import os
import json
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from fastmcp import FastMCP
from jira import JIRA
from datetime import datetime
import logging

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===========================
# JIRA CONNECTION CONFIG
# ===========================

class JiraConfig(BaseModel):
    """JIRA connection configuration"""
    url: str = Field(description="JIRA instance URL")
    username: str = Field(description="JIRA username/email")
    api_token: str = Field(description="JIRA API token")
    project_key: str = Field(description="Default project key")


# Initialize JIRA config from environment
JIRA_CONFIG = JiraConfig(
    url=os.getenv("JIRA_URL", "https://your-domain.atlassian.net"),
    username=os.getenv("JIRA_USERNAME", "your-email@company.com"),
    api_token=os.getenv("JIRA_API_TOKEN", "your-api-token"),
    project_key=os.getenv("JIRA_PROJECT_KEY", "PROC")  # Process Mining project
)


# ===========================
# JIRA CLIENT
# ===========================

def get_jira_client() -> JIRA:
    """Get authenticated JIRA client"""
    try:
        jira = JIRA(
            server=JIRA_CONFIG.url,
            basic_auth=(JIRA_CONFIG.username, JIRA_CONFIG.api_token),
            options={'verify': True}
        )
        return jira
    except Exception as e:
        logger.error(f"Failed to connect to JIRA: {e}")
        raise


# ===========================
# PYDANTIC MODELS
# ===========================

class CreateTicketInput(BaseModel):
    """Input for creating a JIRA ticket - simplified to 3 essential fields"""
    title: str = Field(description="Ticket title/summary")
    description: str = Field(description="Detailed description")
    priority: str = Field(default="Medium", description="Priority: Low, Medium, High, Critical")


class SearchTicketsInput(BaseModel):
    """Input for searching JIRA tickets"""
    jql: str = Field(description="JQL query string")
    max_results: int = Field(default=50, description="Maximum number of results")
    fields: Optional[List[str]] = Field(default=None, description="Fields to return")


class UpdateTicketInput(BaseModel):
    """Input for updating a JIRA ticket"""
    ticket_key: str = Field(description="JIRA ticket key (e.g., PROC-123)")
    fields: Dict[str, Any] = Field(description="Fields to update")
    comment: Optional[str] = Field(default=None, description="Comment to add")


class TicketInfo(BaseModel):
    """JIRA ticket information"""
    key: str
    summary: str
    status: str
    priority: str
    assignee: Optional[str]
    created: str
    updated: str
    description: str
    labels: List[str]
    url: str


class CreateTicketOutput(BaseModel):
    """Output for ticket creation"""
    success: bool
    ticket_key: Optional[str] = None
    ticket_url: Optional[str] = None
    message: str


class SearchTicketsOutput(BaseModel):
    """Output for ticket search"""
    success: bool
    tickets: List[TicketInfo]
    total_count: int
    message: str


# ===========================
# MCP SERVER
# ===========================

mcp = FastMCP("jira-server")


@mcp.tool("create_ticket")
def create_ticket(title: str, description: str) -> CreateTicketOutput:
    """
    Create a new JIRA ticket with title and description only.

    Args:
        title: Ticket title/summary
        description: Detailed description

    Returns:
        CreateTicketOutput with creation results
    """
    try:
        jira = get_jira_client()
        project_key = JIRA_CONFIG.project_key  # Use default from config

        # Build issue dict with minimal required fields only
        issue_dict = {
            'project': {'key': project_key},
            'summary': title,
            'description': description,
            'issuetype': {'name': 'Task'}
        }

        # Create the ticket
        new_issue = jira.create_issue(fields=issue_dict)

        # Try to add labels after creation (optional)
        try:
            new_issue.update(fields={'labels': ['process-mining']})
        except:
            logger.info("Labels not added (field may not be available)")

        ticket_url = f"{JIRA_CONFIG.url}/browse/{new_issue.key}"

        logger.info(f"Created JIRA ticket: {new_issue.key}")

        return CreateTicketOutput(
            success=True,
            ticket_key=new_issue.key,
            ticket_url=ticket_url,
            message=f"Successfully created ticket {new_issue.key}"
        )

    except Exception as e:
        logger.error(f"Failed to create JIRA ticket: {e}")
        return CreateTicketOutput(
            success=False,
            message=f"Failed to create ticket: {str(e)}"
        )


@mcp.tool("search_tickets")
def search_tickets(params: SearchTicketsInput) -> SearchTicketsOutput:
    """
    Search for JIRA tickets using JQL.

    Args:
        params: SearchTicketsInput with search criteria

    Returns:
        SearchTicketsOutput with search results
    """
    try:
        jira = get_jira_client()

        # Default fields if not specified
        fields = params.fields or ['summary', 'status', 'priority', 'assignee', 'created', 'updated', 'description',
                                   'labels']

        # Search for issues
        issues = jira.search_issues(
            jql_str=params.jql,
            maxResults=params.max_results,
            fields=fields
        )

        # Convert to TicketInfo objects
        tickets = []
        for issue in issues:
            assignee_name = None
            if hasattr(issue.fields, 'assignee') and issue.fields.assignee:
                assignee_name = issue.fields.assignee.displayName

            labels = getattr(issue.fields, 'labels', [])

            ticket_info = TicketInfo(
                key=issue.key,
                summary=issue.fields.summary,
                status=issue.fields.status.name,
                priority=issue.fields.priority.name if issue.fields.priority else "None",
                assignee=assignee_name,
                created=issue.fields.created,
                updated=issue.fields.updated,
                description=getattr(issue.fields, 'description', '') or '',
                labels=labels,
                url=f"{JIRA_CONFIG.url}/browse/{issue.key}"
            )
            tickets.append(ticket_info)

        return SearchTicketsOutput(
            success=True,
            tickets=tickets,
            total_count=len(tickets),
            message=f"Found {len(tickets)} tickets"
        )

    except Exception as e:
        logger.error(f"Failed to search JIRA tickets: {e}")
        return SearchTicketsOutput(
            success=False,
            tickets=[],
            total_count=0,
            message=f"Search failed: {str(e)}"
        )


@mcp.tool("check_duplicate_tickets")
def check_duplicate_tickets(title: str, days_back: int = 7) -> Dict[str, Any]:
    """
    Check for duplicate tickets with similar title in recent days.

    Args:
        title: Ticket title to search for
        days_back: Number of days to look back (default: 7)

    Returns:
        Dict with duplicate check results
    """
    try:
        jira = get_jira_client()

        # Create JQL to search for similar tickets
        project_key = JIRA_CONFIG.project_key
        jql = f'''
        project = "{project_key}" 
        AND summary ~ "{title}" 
        AND created >= -{days_back}d 
        AND status != "Closed"
        ORDER BY created DESC
        '''

        # Search for potential duplicates
        search_result = search_tickets(SearchTicketsInput(
            jql=jql,
            max_results=10,
            fields=['summary', 'status', 'created', 'key']
        ))

        if search_result.success and search_result.tickets:
            return {
                "has_duplicates": True,
                "duplicate_count": len(search_result.tickets),
                "existing_tickets": [
                    {
                        "key": ticket.key,
                        "summary": ticket.summary,
                        "status": ticket.status,
                        "created": ticket.created,
                        "url": ticket.url
                    }
                    for ticket in search_result.tickets
                ],
                "recommendation": "Consider updating existing ticket instead of creating new one"
            }
        else:
            return {
                "has_duplicates": False,
                "duplicate_count": 0,
                "existing_tickets": [],
                "recommendation": "No duplicates found, safe to create new ticket"
            }

    except Exception as e:
        logger.error(f"Failed to check for duplicates: {e}")
        return {
            "has_duplicates": False,
            "duplicate_count": 0,
            "existing_tickets": [],
            "error": str(e),
            "recommendation": "Duplicate check failed, proceed with caution"
        }


@mcp.tool("update_ticket")
def update_ticket(params: UpdateTicketInput) -> Dict[str, Any]:
    """
    Update an existing JIRA ticket.

    Args:
        params: UpdateTicketInput with update details

    Returns:
        Dict with update results
    """
    try:
        jira = get_jira_client()

        # Get the issue
        issue = jira.issue(params.ticket_key)

        # Update fields
        if params.fields:
            issue.update(fields=params.fields)

        # Add comment if provided
        if params.comment:
            jira.add_comment(issue, params.comment)

        return {
            "success": True,
            "ticket_key": params.ticket_key,
            "message": f"Successfully updated ticket {params.ticket_key}",
            "url": f"{JIRA_CONFIG.url}/browse/{params.ticket_key}"
        }

    except Exception as e:
        logger.error(f"Failed to update ticket {params.ticket_key}: {e}")
        return {
            "success": False,
            "ticket_key": params.ticket_key,
            "message": f"Failed to update ticket: {str(e)}"
        }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="sse", host="127.0.0.1", port=8001)