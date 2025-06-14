# Project Launch Instructions

**IMPORTANT:** The project was run on a **Windows** system.

## Steps to follow

1. **Install PostgreSQL**, create a database and a table with any name.  
   Access credentials (database name, user, password, table name, etc.) should be entered in the `.env` file.  
   There is a sample `.env.example` file in the repository — use it as a template to fill in your actual `.env` file.

2. **Create an account on [Anthropic](https://console.anthropic.com/)** and add funds to it.  
   Otherwise, I can share my `API_KEY` via email.

3. The repository contains example `.bat` launch files with the `_example` suffix.  
   Create your own `.bat` files (without `_example` in the name), filling in the appropriate **project paths**.

4. **Create a virtual environment** and install the required libraries according to `requirements.txt`:
   
`pip install -r requirements.txt`

5. **Download [Claude Desktop](https://claude.ai/download)**.  
Go to `File -> Settings -> Developer -> Edit Config`.  
Paste the contents of the `claude_desktop_config_example.json`, file there, filling in the  **paths to your `.bat` files**.  
> Make sure to use the correct path format.

6. **Extract the file** `BPI Challenge 2017.xes.gz` into the `datasets` folder.

7. **Run the data import script**:

`python import_data.py`

The script should import the event log data into the database.

8. **Restart Claude Desktop**.  
After launching the tool, the tools should be visible below the input field (two sliders labeled "search and tools").
There you can enable or disable the available tools.

## Optional: Configuring MCP in Cursor

If you are using **[Cursor](https://www.cursor.com/)** with MCP tools:

1. Go to `File -> Preferences -> Cursor Settings`. 
2. Go to `MCP Tools` tab.
3. Press `New MCP Server button` and paste previously filled `claude_desktop_config.json` file contents, adjusting the **paths to your `.bat` files** as needed.
4. Save the configuration. Your MCP tools should now be available inside Cursor.