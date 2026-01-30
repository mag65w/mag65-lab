from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from github import Github
import os
import json
import uuid
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use GitHub Enterprise URL if set, otherwise use default GitHub.com
github_token = os.environ["GITHUB_TOKEN"]
github_enterprise_url = os.environ.get("GITHUB_ENTERPRISE_URL")

if github_enterprise_url:
    # Use GitHub Enterprise
    # Automatically add /api/v3 if not included in URL
    base_url = github_enterprise_url.rstrip("/")
    if not base_url.endswith("/api/v3"):
        base_url = f"{base_url}/api/v3"
    gh = Github(base_url=base_url, login_or_token=github_token)
else:
    # Use default GitHub.com
    gh = Github(github_token)

# Create FastAPI app
app = FastAPI()

# Function to retrieve GitHub repository information
def get_repository_info(org: str, repo: str) -> Dict[str, Any]:
    """Retrieve GitHub repository information"""
    try:
        # First verify that the token is properly authenticated
        try:
            user = gh.get_user()
            user_login = user.login
        except Exception as auth_error:
            raise Exception(f"Authentication failed: {str(auth_error)}")
        
        # Query repository in org/repo format
        repo_full_name = f"{org}/{repo}"
        r = gh.get_repo(repo_full_name)
        
        return {
            "description": r.description,
            "stars": r.stargazers_count,
            "language": r.language,
            "open_issues": r.open_issues_count
        }
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "Not Found" in error_msg:
            raise Exception(f"Repository '{org}/{repo}' not found. Please check if the repository exists and you have access to it.")
        elif "401" in error_msg or "Unauthorized" in error_msg or "Bad credentials" in error_msg:
            raise Exception(f"Authentication failed or insufficient permissions. Please check your GITHUB_TOKEN.")
        else:
            raise Exception(f"Error accessing repository: {error_msg}")

# MCP protocol endpoint
@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """MCP protocol HTTP endpoint"""
    try:
        body = await request.json()
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")
        
        if method == "initialize":
            # Initialize session
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "github-mcp-server",
                        "version": "1.0.0"
                    }
                }
            }
            return Response(content=json.dumps(response), media_type="application/json")
        
        elif method == "tools/list":
            # Return list of available tools
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "summarize_repository",
                            "description": "Summarize a GitHub repository from an organization",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "org": {
                                        "type": "string",
                                        "description": "Organization name"
                                    },
                                    "repo": {
                                        "type": "string",
                                        "description": "Repository name"
                                    }
                                },
                                "required": ["org", "repo"]
                            }
                        }
                    ]
                }
            }
            return Response(content=json.dumps(response), media_type="application/json")
        
        elif method == "tools/call":
            # Execute tool
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "summarize_repository":
                org = arguments.get("org")
                repo = arguments.get("repo")
                
                if not org or not repo:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32602,
                            "message": "Invalid params",
                            "data": "org and repo are required"
                        }
                    }
                    return Response(content=json.dumps(error_response), media_type="application/json", status_code=400)
                
                try:
                    result = get_repository_info(org, repo)
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(result, indent=2, ensure_ascii=False)
                                }
                            ]
                        }
                    }
                    return Response(content=json.dumps(response), media_type="application/json")
                except Exception as e:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32000,
                            "message": str(e)
                        }
                    }
                    return Response(content=json.dumps(error_response), media_type="application/json", status_code=500)
            else:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {tool_name}"
                    }
                }
                return Response(content=json.dumps(error_response), media_type="application/json", status_code=404)
        
        else:
            error_response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
            return Response(content=json.dumps(error_response), media_type="application/json", status_code=404)
    
    except json.JSONDecodeError:
        error_response = {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32700,
                "message": "Parse error"
            }
        }
        return Response(content=json.dumps(error_response), media_type="application/json", status_code=400)
    except Exception as e:
        error_response = {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }
        return Response(content=json.dumps(error_response), media_type="application/json", status_code=500)

# Legacy REST endpoint (backward compatibility)
@app.post("/mcp/tools/summarize_repository")
async def summarize_repository(org: str, repo: str):
    """Summarize a GitHub repository from an organization (REST API)
    
    Args:
        org: Organization name
        repo: Repository name
    """
    try:
        result = get_repository_info(org, repo)
        return result
    except Exception as e:
        error_msg = str(e)
        enterprise_info = f" (Enterprise: {github_enterprise_url})" if github_enterprise_url else ""
        
        if "not found" in error_msg.lower():
            raise HTTPException(status_code=404, detail=f"{error_msg}{enterprise_info}")
        elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            raise HTTPException(status_code=401, detail=f"{error_msg}{enterprise_info}")
        else:
            raise HTTPException(status_code=500, detail=f"{error_msg}{enterprise_info}")

