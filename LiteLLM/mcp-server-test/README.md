# GitHub MCP Server

A FastAPI server implementing MCP (Model Context Protocol) for querying GitHub repository information.

## Features

- GitHub repository summary information retrieval
  - Repository description
  - Star count
  - Primary programming language
  - Open issues count

## Prerequisites

- Python 3.8 or higher
- GitHub Personal Access Token

## Installation

1. Clone or download the repository

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp env.sample .env
```

5. Open the `.env` file and configure the following:
   - `GITHUB_TOKEN`: GitHub Personal Access Token
   - `GITHUB_ENTERPRISE_URL` (optional): Domain URL if using GitHub Enterprise

## GitHub Enterprise Configuration

If using GitHub Enterprise, set `GITHUB_ENTERPRISE_URL` in the `.env` file:

```bash
GITHUB_ENTERPRISE_URL=https://github.enterprise.com
```

Or you can include the API path:

```bash
GITHUB_ENTERPRISE_URL=https://github.enterprise.com/api/v3
```

If `/api/v3` is not included, it will be automatically added.

**Note**: If `GITHUB_ENTERPRISE_URL` is not set, the default GitHub.com will be used.

## GitHub Token Generation

**Important**: This server uses **Classic tokens**. Fine-grained tokens may not work properly with GitHub Enterprise.

### For GitHub.com

1. Log in to GitHub
2. Click your profile icon in the top right > **Settings**
3. Click **Developer settings** in the left menu
4. Select **Personal access tokens** > **Tokens (classic)**
5. Click **Generate new token (classic)**
6. Enter a token name (e.g., "MCP Server")
7. Select required permissions:
   - `public_repo` (read access to public repositories)
   - or `repo` (full access to all repositories)
8. Click **Generate token**
9. Copy the generated token and paste it into the `.env` file

### For GitHub Enterprise

**Important**: GitHub Enterprise requires **Classic tokens**. Fine-grained tokens may not be supported.

1. Log in to your GitHub Enterprise instance
2. Click your profile icon in the top right > **Settings**
3. Click **Developer settings** in the left menu
4. Select **Personal access tokens** > **Tokens (classic)**
5. Click **Generate new token (classic)**
6. Enter a token name (e.g., "MCP Server")
7. Select required permissions (may vary by organization policy):
   - `repo` (full access to all repositories)
8. Click **Generate token**
9. Copy the generated token and paste it into `GITHUB_TOKEN` in the `.env` file
10. Also set `GITHUB_ENTERPRISE_URL` in the `.env` file

## Running the Server

To run the server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Once the server is running, the MCP endpoint will be available at `/mcp`.

## API Usage Example

Once the server is running, you can use the `summarize_repository` tool through an MCP client.

### Parameters
- `org`: GitHub organization name (e.g., "mycompany")
- `repo`: Repository name (e.g., "my-project")

### Response Example
```json
{
  "description": "My first repository on GitHub!",
  "stars": 1234,
  "language": "Python",
  "open_issues": 5
}
```

## License

This project does not have a license specified.

## API Request Example

Example curl request to use the MCP server:

```bash
curl --location 'https://your-api-endpoint.com/v1/responses' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer YOUR_API_KEY' \
--data '{
    "model": "your-model-name",
    "input": [
        {
            "role": "user",
            "content": "mycompany organization's myproject repository",
            "type": "message"
        }
    ],
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy/mcp",
            "require_approval": "never",
            "allowed_tools": ["summarize_repository"]
        }
    ],
    "stream": false,
    "tool_choice": "required"
}'
```
