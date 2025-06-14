import os
import random
from contextlib import contextmanager
from dotenv import load_dotenv
from fastmcp import FastMCP
from pandas import DataFrame
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

import pm4py

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from config import DB_CONFIG, TABLE_SCHEMA, DatabaseConfig, TableSchema

# Load environment variables
load_dotenv()

# ===========================
# DATABASE CONNECTION SCHEMAS
# ===========================


class LogSource(BaseModel):
    # For database sources
    db_config: Optional[DatabaseConfig] = Field(None, description="Database configuration")
    table_schema: Optional[TableSchema] = Field(None, description="Event log table schema")

    # Caching options
    use_cache: bool = Field(default=True, description="Cache loaded event logs")
    cache_key: Optional[str] = Field(None, description="Custom cache key")


# ===========================
# DATABASE UTILITIES
# ===========================

class EventLogCache:
    """Simple in-memory cache for event logs"""
    _cache: Dict[str, DataFrame] = {}

    @classmethod
    def get(cls, key: str) -> Optional[DataFrame]:
        return cls._cache.get(key)

    @classmethod
    def set(cls, key: str, log: DataFrame):
        cls._cache[key] = log

    @classmethod
    def clear(cls):
        cls._cache.clear()


@contextmanager
def get_db_connection(config: DatabaseConfig):
    """Context manager for database connections"""
    engine = create_engine(config.connection_string)
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()
        engine.dispose()


def build_event_log_query(schema: TableSchema, db_schema: str = "public") -> str:
    """Build SQL query for extracting event log"""

    # Base columns
    columns = [
        f"{schema.case_column} as \"case:concept:name\"",
        f"{schema.activity_column} as \"concept:name\"",
        f"{schema.timestamp_column} as \"time:timestamp\""
    ]

    # Add resource column if specified
    if schema.resource_column:
        columns.append(f"{schema.resource_column} as \"org:resource\"")

    # Add additional columns
    if schema.additional_columns:
        columns.extend(schema.additional_columns)

    # Build query
    query = f"""
    SELECT {', '.join(columns)}
    FROM {db_schema}.{schema.table_name}
    """

    # Add WHERE clause
    if schema.where_clause:
        query += f"\nWHERE {schema.where_clause}"

    return query


def load_from_database(config: DatabaseConfig, schema: TableSchema) -> DataFrame:
    """Load event log from database"""

    # Build query
    query = build_event_log_query(schema, config.db_schema)

    # Execute query and load into DataFrame
    with get_db_connection(config) as conn:
        df = pd.read_sql(text(query), conn)

    # Ensure timestamp column is datetime
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)

    # Convert to PM4Py event log
    return df


# ===========================
# ENHANCED LOAD FUNCTION
# ===========================

def load_event_log(source: LogSource) -> DataFrame:
    """
    Load event log from database.
    """

    # Generate cache key
    if source.use_cache:
        if source.cache_key:
            cache_key = source.cache_key
        else:
            cache_key = f"db:{source.db_config.database}:{source.table_schema.table_name}"

        # Check cache
        cached_log = EventLogCache.get(cache_key)
        if cached_log is not None:
            return cached_log

    df = load_from_database(source.db_config, source.table_schema)

    # Cache if enabled
    if source.use_cache:
        EventLogCache.set(cache_key, df)

    return df


mcp = FastMCP("process-mining-server")

table_name: str = Field(description="Name of the event log table")
case_column: str = Field(description="Column containing case IDs")
activity_column: str = Field(description="Column containing activity names")
timestamp_column: str = Field(description="Column containing timestamps")

log_source = LogSource(
    db_config=DB_CONFIG,
    table_schema=TABLE_SCHEMA,
    cache_key=f"db:{TABLE_SCHEMA.table_name}",
)



class BasicStatsOutput(BaseModel):
    """Output for basic statistics analysis"""
    total_cases: int
    total_events: int
    total_activities: int
    avg_events_per_case: float
    log_timeframe: Dict[str, Any]
    case_duration_stats: Dict[str, float]
    resource_stats: Optional[Dict[str, Any]] = None
    activity_frequencies: Optional[Dict[str, float]] = None


@mcp.tool("get_basic_stats")
def get_basic_stats() -> BasicStatsOutput:
    """
    Get basic statistics about an event log.

    Returns fundamental metrics including:
    - Total number of cases, events, and unique activities
    - Average events per case
    - Log timeframe (start, end, duration)
    - Case duration statistics (mean, median, min, max)
    - Resource utilization patterns
    - Activity frequencies (relative frequencies for all activities)
    """
    df = load_event_log(log_source)

    # Calculate case durations
    case_durations = df.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max'])
    case_durations['duration'] = (case_durations['max'] - case_durations['min']).dt.total_seconds() / 3600  # hours

    # Basic stats
    basic_stats = {
        "total_cases": df['case:concept:name'].nunique(),
        "total_events": len(df),
        "total_activities": df['concept:name'].nunique(),
        "avg_events_per_case": len(df) / df['case:concept:name'].nunique(),
        "log_timeframe": {
            "start": df['time:timestamp'].min().isoformat(),
            "end": df['time:timestamp'].max().isoformat(),
            "duration_days": (df['time:timestamp'].max() - df['time:timestamp'].min()).days
        },
        "case_duration_stats": {
            "mean_hours": float(case_durations['duration'].mean()),
            "median_hours": float(case_durations['duration'].median()),
            "std_hours": float(case_durations['duration'].std()),
            "min_hours": float(case_durations['duration'].min()),
            "max_hours": float(case_durations['duration'].max())
        }
    }

    # Resource statistics (if resource column exists)
    if 'org:resource' in df.columns:
        resource_stats = df.groupby('org:resource').agg({
            'case:concept:name': 'nunique',
            'concept:name': 'count'
        })
        basic_stats["resource_stats"] = {
            "total_resources": len(resource_stats),
            "avg_events_per_resource": float(resource_stats['concept:name'].mean()),
            "avg_cases_per_resource": float(resource_stats['case:concept:name'].mean()),
            "resource_utilization": {
                "min_cases": int(resource_stats['case:concept:name'].min()),
                "max_cases": int(resource_stats['case:concept:name'].max()),
                "min_events": int(resource_stats['concept:name'].min()),
                "max_events": int(resource_stats['concept:name'].max())
            }
        }

    # Activity frequencies
    activity_freq = df['concept:name'].value_counts(normalize=True) * 100  # Convert to percentages
    basic_stats["activity_frequencies"] = activity_freq.to_dict()  # All activities with their relative frequencies

    return BasicStatsOutput(**basic_stats)


class ProcessDiscoveryInput(BaseModel):
    """Input for process discovery"""
    discovery_type: Literal["dfg", "petri_net"] = Field(
        default="petri_net",
        description="Type of process model to discover"
    )
    discovery_algorithm: Optional[Literal["inductive", "alpha", "heuristic"]] = Field(
        default="inductive",
        description="Algorithm to use for Petri net discovery (only used if discovery_type is petri_net)"
    )
    save_model: bool = Field(
        default=False,
        description="Save the model to a file"
    )


class ProcessInfo(BaseModel):
    """Information about discovered process model"""
    model_type: str
    model_description: str
    model_stats: Dict[str, Any]


class ProcessDiscoveryOutput(BaseModel):
    """Output for process discovery"""
    process_info: ProcessInfo


@mcp.tool("discover_process")
def discover_process(params: ProcessDiscoveryInput) -> ProcessDiscoveryOutput:
    """
    Discover a process model from the event log.

    Args:
        params: ProcessDiscoveryInput containing:
            - discovery_type: Type of model to discover ("dfg" or "petri_net")
            - discovery_algorithm: Algorithm to use for Petri net discovery (only for petri_net)

    Returns:
        ProcessDiscoveryOutput containing:
            - process_info: Information about the discovered model including:
                - model_type: Type of discovered model
                - model_description: Natural language description of the model
                - model_stats: Statistics about the model
    """
    df = load_event_log(log_source)
    log = pm4py.convert_to_event_log(df)

    # Create output directory if it doesn't exist
    output_dir = "process_models"
    os.makedirs(output_dir, exist_ok=True)

    if params.discovery_type == "dfg":
        # Discover DFG
        dfg = pm4py.discover_dfg(log)
        # Get DFG abstraction using LLM
        model_description = pm4py.llm.abstract_dfg(log)

        # Calculate DFG statistics
        model_stats = {
            "num_activities": len(dfg[0]),
            "num_arcs": len(dfg[1]),
            "max_frequency": max(dfg[1].values()) if dfg[1] else 0,
            "min_frequency": min(dfg[1].values()) if dfg[1] else 0
        }

        # Save DFG visualization
        if params.save_model:
            from pm4py.visualization.dfg import visualizer as dfg_visualizer
            gviz = dfg_visualizer.apply(dfg[0], dfg[1], log=log, variant=dfg_visualizer.Variants.FREQUENCY)
            dfg_visualizer.save(gviz, os.path.join(output_dir, "dfg_model.png"))

        process_info = ProcessInfo(
            model_type="directly-follows graph",
            model_description=model_description,
            model_stats=model_stats
        )

    else:  # petri_net
        # Discover Petri net
        if params.discovery_algorithm == "inductive":
            net, im, fm = pm4py.discover_petri_net_inductive(log)
        elif params.discovery_algorithm == "alpha":
            net, im, fm = pm4py.discover_petri_net_alpha(log)
        else:  # heuristic
            net, im, fm = pm4py.discover_petri_net_heuristics(log)

        # Get Petri net abstraction using LLM
        model_description = pm4py.llm.abstract_petri_net(net, im, fm)

        # Calculate Petri net statistics
        model_stats = {
            "places": len(net.places),
            "transitions": len(net.transitions),
            "arcs": len(net.arcs),
            "algorithm": params.discovery_algorithm
        }

        # Save Petri net visualization
        if params.save_model:
            from pm4py.visualization.petri_net import visualizer as pn_visualizer
            gviz = pn_visualizer.apply(net, im, fm, variant=pn_visualizer.Variants.WO_DECORATION)
            pn_visualizer.save(gviz, os.path.join(output_dir, "petri_net_model.png"))

        process_info = ProcessInfo(
            model_type="petri net",
            model_description=model_description,
            model_stats=model_stats,
        )

    return ProcessDiscoveryOutput(process_info=process_info)

@mcp.tool("get_process_variants_summary")
def get_process_variants_summary() -> Dict[str, str]:
    """
    Get a natural language summary of process variants.
    """
    df = load_event_log(log_source)
    log = pm4py.convert_to_event_log(df)

    return {
        "summary": pm4py.llm.abstract_variants(log),
        "total_variants": len(pm4py.get_variants(log))
    }


class ConformanceCheckInput(BaseModel):
    """Input for conformance checking"""
    case_id: str
    discovery_algorithm: Optional[Literal["inductive", "alpha", "heuristic"]] = Field(
        default="inductive",
        description="Algorithm to use for Petri net discovery (only used if reference_model_type is petri_net)"
    )


@mcp.tool("check_partial_conformance")
def check_partial_conformance(params: ConformanceCheckInput) -> bool:
    """
    Check if a partial trace conforms to the discovered process model.

    Discovers a process model from the event log and checks if the current trace
    for the specified case follows the expected process flow using token-based replay.

    Args:
        params: ConformanceCheckInput with case_id and discovery_algorithm

    Returns:
        bool: True if conformant (no missing tokens), False if violations detected
    """
    random.seed(112)
    case_id = params.case_id

    df = load_event_log(log_source)

    case_trace = df[df["case:concept:name"] == case_id].copy()
    case_trace = case_trace.reset_index(drop=True)

    df = df[df["case:concept:name"] != case_id].copy()

    if params.discovery_algorithm == "inductive":
        net, im, fm = pm4py.discover_petri_net_inductive(df)
    elif params.discovery_algorithm == "alpha":
        net, im, fm = pm4py.discover_petri_net_alpha(df)
    else:  # heuristic
        net, im, fm = pm4py.discover_petri_net_heuristics(df)

    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    # Calculate fitness using token-based replay
    replayed_traces = token_replay.apply(case_trace, net, im, fm)

    result = replayed_traces[0]

    return result["missing_tokens"] == 0

class PerformanceAnalysisInput(BaseModel):
    """Input for performance analysis"""
    bottleneck_threshold: float = Field(
        default=1.5,
        description="Threshold for bottleneck detection (times the median duration)"
    )
    percentage_of_cases: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Percentage of cases to analyze"
    )


class ActivityTiming(BaseModel):
    """Timing information for an activity"""
    name: str
    avg_duration: float  # in hours
    median_duration: float  # in hours
    min_duration: float  # in hours
    max_duration: float  # in hours
    std_duration: float  # in hours
    is_bottleneck: bool
    waiting_time: Optional[float] = None  # in hours
    frequency: int


class PathTiming(BaseModel):
    """Timing information for a path"""
    path: List[str]
    avg_duration: float  # in hours
    median_duration: float  # in hours
    frequency: int
    bottleneck_activities: List[str]


class PerformanceAnalysisOutput(BaseModel):
    """Output for performance analysis"""
    case_duration_stats: Dict[str, float]
    activity_timings: List[ActivityTiming]
    bottleneck_analysis: Dict[str, Any]


@mcp.tool("analyze_performance")
def analyze_performance(params: PerformanceAnalysisInput) -> PerformanceAnalysisOutput:
    """
    Analyze process performance including timing and bottlenecks.

    Args:
        params: PerformanceAnalysisInput containing:
            - bottleneck_threshold: Threshold for bottleneck detection
            - percentage_of_cases: Percentage of cases to analyze

    Returns:
        PerformanceAnalysisOutput containing:
            - case_duration_stats: Statistics about case durations
            - activity_timings: Detailed timing information for each activity
            - bottleneck_analysis: Analysis of bottlenecks
    """
    random.seed(112)
    df = load_event_log(log_source)

    # Sample cases if needed
    if params.percentage_of_cases < 1.0:
        case_ids = df["case:concept:name"].unique()
        sampled_cases = random.sample(list(case_ids), int(len(case_ids) * params.percentage_of_cases))
        df = df[df["case:concept:name"].isin(sampled_cases)]

    # Convert timestamps to datetime if they aren't already
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

    # Calculate case durations
    case_durations = df.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max'])
    case_durations['duration'] = (case_durations['max'] - case_durations['min']).dt.total_seconds() / 3600  # hours

    # Calculate activity durations
    df['next_timestamp'] = df.groupby('case:concept:name')['time:timestamp'].shift(-1)
    df['activity_duration'] = (df['next_timestamp'] - df['time:timestamp']).dt.total_seconds() / 3600  # hours

    # Analyze activity timings
    activity_stats = df.groupby('concept:name').agg({
        'activity_duration': ['mean', 'median', 'min', 'max', 'std', 'count']
    })

    # Calculate bottleneck threshold
    median_duration = activity_stats[('activity_duration', 'median')].median()
    bottleneck_threshold = median_duration * params.bottleneck_threshold

    # Prepare activity timings
    activity_timings = []
    for activity in activity_stats.index:
        stats = activity_stats.loc[activity]
        activity_timings.append(ActivityTiming(
            name=activity,
            avg_duration=float(stats[('activity_duration', 'mean')]),
            median_duration=float(stats[('activity_duration', 'median')]),
            min_duration=float(stats[('activity_duration', 'min')]),
            max_duration=float(stats[('activity_duration', 'max')]),
            std_duration=float(stats[('activity_duration', 'std')]),
            is_bottleneck=float(stats[('activity_duration', 'median')]) > bottleneck_threshold,
            frequency=int(stats[('activity_duration', 'count')])
        ))

    # Prepare bottleneck analysis
    bottleneck_activities = [a for a in activity_timings if a.is_bottleneck]
    bottleneck_analysis = {
        "total_bottlenecks": len(bottleneck_activities),
        "bottleneck_activities": [a.name for a in bottleneck_activities],
        "bottleneck_impact": {
            "avg_delay": float(sum(a.avg_duration for a in bottleneck_activities) / len(
                bottleneck_activities)) if bottleneck_activities else 0.0,
            "max_delay": float(max(a.max_duration for a in bottleneck_activities)) if bottleneck_activities else 0.0,
            "affected_cases": int(sum(a.frequency for a in bottleneck_activities))
        }
    }

    return PerformanceAnalysisOutput(
        case_duration_stats={
            "mean": float(case_durations['duration'].mean()),
            "median": float(case_durations['duration'].median()),
            "min": float(case_durations['duration'].min()),
            "max": float(case_durations['duration'].max()),
            "std": float(case_durations['duration'].std())
        },
        activity_timings=activity_timings,
        bottleneck_analysis=bottleneck_analysis
    )


class VariantAnalysisInput(BaseModel):
    """Input for variant analysis"""
    top_k: int = Field(default=10, ge=1, le=100, description="Number of top variants to return")
    coverage_threshold: float = Field(default=0.8, ge=0.0, le=1.0,
                                      description="Calculate how many variants cover this percentage of cases")
    percentage_of_cases: float = Field(default=1.0, ge=0.0, le=1.0,
                                       description="Percentage of cases to analyze")


class VariantInfo(BaseModel):
    """Information about a process variant"""
    rank: int
    trace: tuple
    count: int
    percentage: float
    cumulative_percentage: float
    num_activities: int


class VariantAnalysisOutput(BaseModel):
    """Output for variant analysis"""
    total_variants: int
    top_variants: List[VariantInfo]
    coverage_analysis: Dict[str, Any]
    variant_complexity: Dict[str, float]


@mcp.tool("analyze_variants")
def analyze_variants(params: VariantAnalysisInput) -> VariantAnalysisOutput:
    """
    Analyze process variants (unique paths through the process).

    Args:
        params: VariantAnalysisInput containing:
            - top_k: Number of top variants to return
            - coverage_threshold: Threshold for coverage analysis
            - percentage_of_cases: Percentage of cases to analyze

    Returns:
        VariantAnalysisOutput containing:
            - total_variants: Total number of unique variants
            - top_variants: Details of top K most frequent variants
            - coverage_analysis: Coverage analysis results
            - variant_complexity: Variant complexity metrics
    """
    random.seed(113)
    df = load_event_log(log_source)

    # Sample cases if needed
    if params.percentage_of_cases < 1.0:
        case_ids = df["case:concept:name"].unique()
        sampled_cases = random.sample(list(case_ids), int(len(case_ids) * params.percentage_of_cases))
        df = df[df["case:concept:name"].isin(sampled_cases)]

    log = pm4py.convert_to_event_log(df)

    # Get variants directly from the log
    variants = pm4py.get_variants(log)
    variant_counts = {variant: len(cases) for variant, cases in variants.items()}

    # Sort variants by count
    variants_sorted = sorted(variant_counts.items(), key=lambda x: x[1], reverse=True)
    total_cases = sum(count for _, count in variants_sorted)

    # Find variants covering threshold percentage
    cumulative_count = 0
    threshold_variants = []
    for variant, count in variants_sorted:
        cumulative_count += count
        threshold_variants.append((variant, count))
        if cumulative_count / total_cases >= params.coverage_threshold:
            break

    # Build top variants list
    top_variants = []
    cumulative = 0
    for i, (variant, count) in enumerate(variants_sorted[:params.top_k]):
        cumulative += count
        # Convert variant to tuple of activities
        trace = tuple(variant)
        top_variants.append(VariantInfo(
            rank=i + 1,
            trace=trace,
            count=count,
            percentage=float(count / total_cases * 100),
            cumulative_percentage=float(cumulative / total_cases * 100),
            num_activities=len(trace)
        ))

    # Calculate variant complexity metrics
    variant_lengths = [len(variant) for variant, _ in variants_sorted]

    return VariantAnalysisOutput(
        total_variants=len(variants_sorted),
        top_variants=top_variants,
        coverage_analysis={
            "variants_for_threshold": len(threshold_variants),
            "threshold_percentage": params.coverage_threshold * 100,
            "percentage_of_total_variants": float(len(threshold_variants) / len(variants_sorted) * 100)
        },
        variant_complexity={
            "mean_length": float(np.mean(variant_lengths)),
            "median_length": float(np.median(variant_lengths)),
            "min_length": int(np.min(variant_lengths)),
            "max_length": int(np.max(variant_lengths))
        }
    )

class ActivityAnalysisInput(BaseModel):
    """Input for activity analysis"""
    top_k: int = Field(default=10, ge=1, le=100, description="Number of top activities to analyze in detail")
    include_statistics: bool = Field(default=True, description="Include distribution statistics")


class ActivityInfo(BaseModel):
    """Information about a single activity"""
    name: str
    frequency: int
    percentage: float
    cases_involved: int
    first_occurrence: str
    last_occurrence: str


class ActivityAnalysisOutput(BaseModel):
    """Output for activity analysis"""
    unique_activities: int
    top_activities: List[ActivityInfo]
    distribution_stats: Optional[Dict[str, float]] = None


@mcp.tool("analyze_activities")
def analyze_activities(params: ActivityAnalysisInput) -> ActivityAnalysisOutput:
    """
    Analyze activities in the event log.

    Provides:
    - Count of unique activities
    - Detailed information about top K activities
    - Activity frequency distribution statistics
    """

    df = load_event_log(log_source)

    # Activity frequency
    activity_freq = df['concept:name'].value_counts()

    # Activity statistics
    activity_stats = df.groupby('concept:name').agg({
        'case:concept:name': 'nunique',
        'time:timestamp': ['count', 'min', 'max']
    })

    # Build top activities list
    top_activities = []
    for activity in activity_freq.head(params.top_k).index:
        top_activities.append(ActivityInfo(
            name=activity,
            frequency=int(activity_freq[activity]),
            percentage=float(activity_freq[activity] / len(df) * 100),
            cases_involved=int(activity_stats.loc[activity, ('case:concept:name', 'nunique')]),
            first_occurrence=activity_stats.loc[activity, ('time:timestamp', 'min')].isoformat(),
            last_occurrence=activity_stats.loc[activity, ('time:timestamp', 'max')].isoformat()
        ))

    output = ActivityAnalysisOutput(
        unique_activities=len(activity_freq),
        top_activities=top_activities
    )

    if params.include_statistics:
        output.distribution_stats = {
            "mean": float(activity_freq.mean()),
            "std": float(activity_freq.std()),
            "min": int(activity_freq.min()),
            "max": int(activity_freq.max())
        }

    return output

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    #SSE Transport
    #mcp.run(transport="sse", host="127.0.0.1", port=8000)