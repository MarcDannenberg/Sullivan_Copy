"""
Games API for Sully cognitive system

This module provides FastAPI endpoints for Sully's games functionality,
including visual gameplay and AI opponent capabilities.
"""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import base64
import tempfile
import os

# Import Sully components
from sully import Sully
from sully_engine.kernel_modules.games import SullyGames

# Create a router for games endpoints
games_router = APIRouter(prefix="/games", tags=["games"])

# Initialize Sully games module
sully = Sully()
games_module = SullyGames(reasoning_node=sully.reasoning_node, memory_system=sully.memory)

# Models for request/response
class CreateGameRequest(BaseModel):
    game_type: str  # "mahjong", "chess", or "go"
    player_names: List[str]
    session_id: Optional[str] = "default"
    board_size: Optional[int] = 19  # For Go
    sully_plays_as: Optional[str] = None  # "white" or "black" for Chess/Go, seat position for Mahjong
    difficulty: Optional[str] = "medium"  # "easy", "medium", "hard", "expert"

class MoveRequest(BaseModel):
    move: Dict[str, Any]
    session_id: Optional[str] = "default"
    animate: Optional[bool] = False  # Whether to animate the move in visualization

class SessionRequest(BaseModel):
    session_id: Optional[str] = "default"

class GameStateRequest(BaseModel):
    game_state: Dict[str, Any]
    game_type: str

class RenderRequest(BaseModel):
    session_id: Optional[str] = "default"
    format: Optional[str] = "svg"  # "svg", "png", "html", "json"
    include_hints: Optional[bool] = False  # Whether to include move hints
    highlight_last_move: Optional[bool] = True  # Whether to highlight the last move
    theme: Optional[str] = "default"  # Visual theme for rendering

class ThoughtProcessRequest(BaseModel):
    session_id: Optional[str] = "default"
    depth: Optional[int] = 3  # How detailed Sully's thought explanation should be

# Game creation endpoint
@games_router.post("/create")
async def create_game(request: CreateGameRequest):
    """Create a new game instance"""
    result = games_module.create_game(
        game_type=request.game_type,
        player_names=request.player_names,
        session_id=request.session_id,
        board_size=request.board_size,
        sully_plays_as=request.sully_plays_as,
        difficulty=request.difficulty
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to create game"))
    
    return result

# Making a move
@games_router.post("/move")
async def make_move(request: MoveRequest):
    """Make a move in the current game"""
    result = games_module.make_move(
        move=request.move,
        session_id=request.session_id,
        animate=request.animate
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Invalid move"))
    
    # If it's Sully's turn after this move, automatically generate Sully's move
    if result.get("current_player") == "Sully" and not result.get("game_over", False):
        sully_move_result = games_module.get_sully_move(session_id=request.session_id)
        if sully_move_result["success"]:
            result["sully_move"] = sully_move_result["move"]
            result["sully_reasoning"] = sully_move_result.get("reasoning", "")
            result["state"] = sully_move_result["state"]  # Update with latest state after Sully's move
    
    return result

# Get Sully's move
@games_router.post("/sully_move")
async def get_sully_move(request: SessionRequest):
    """Get Sully's next move in the current game"""
    result = games_module.get_sully_move(session_id=request.session_id)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to generate move"))
    
    return result

# Get Sully's thought process for a move
@games_router.post("/thought_process")
async def get_thought_process(request: ThoughtProcessRequest):
    """Get Sully's detailed thought process for deciding a move"""
    try:
        result = games_module.get_move_thought_process(
            session_id=request.session_id,
            depth=request.depth
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to generate thought process"))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating thought process: {str(e)}")

# Get game state
@games_router.post("/state")
async def get_game_state(request: SessionRequest):
    """Get the current state of the game"""
    result = games_module.get_game_state(session_id=request.session_id)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "No active game"))
    
    return result

# Render visual representation of the game
@games_router.post("/render")
async def render_game(request: RenderRequest):
    """Get a visual representation of the current game state"""
    try:
        result = games_module.render_game(
            session_id=request.session_id,
            format=request.format,
            include_hints=request.include_hints,
            highlight_last_move=request.highlight_last_move,
            theme=request.theme
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to render game"))
        
        # Handle different output formats
        if request.format == "svg":
            return Response(content=result["content"], media_type="image/svg+xml")
        elif request.format == "png":
            # For PNG, we return base64
            return JSONResponse({"image": result["content"], "mime_type": "image/png"})
        elif request.format == "html":
            return Response(content=result["content"], media_type="text/html")
        else:
            # Default to JSON
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rendering game: {str(e)}")

# Get PNG screenshot of the game
@games_router.post("/screenshot")
async def get_screenshot(request: RenderRequest):
    """Get a PNG screenshot of the current game"""
    try:
        result = games_module.render_game(
            session_id=request.session_id,
            format="png",
            include_hints=request.include_hints,
            highlight_last_move=request.highlight_last_move,
            theme=request.theme
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to generate screenshot"))
        
        # Create a temporary file for the image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_file:
            # Decode base64 image
            img_data = base64.b64decode(result["content"])
            img_file.write(img_data)
            img_path = img_file.name
        
        try:
            # Return the file
            return FileResponse(
                img_path, 
                media_type="image/png",
                background=None,
                filename=f"game_{request.session_id}.png"
            )
        finally:
            # Ensure the temporary file is deleted after response is sent
            os.unlink(img_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating screenshot: {str(e)}")

# Get valid moves for current player
@games_router.post("/valid_moves")
async def get_valid_moves(request: SessionRequest):
    """Get all valid moves for the current player"""
    try:
        result = games_module.get_valid_moves(session_id=request.session_id)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to get valid moves"))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting valid moves: {str(e)}")

# Analyze a game state
@games_router.post("/analyze")
async def analyze_game(request: GameStateRequest):
    """Analyze a game state and provide insights"""
    result = games_module.analyze_game(
        game_state=request.game_state,
        game_type=request.game_type
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Analysis failed"))
    
    return result

# Get game suggestion/hint
@games_router.post("/hint")
async def get_hint(request: SessionRequest):
    """Get a hint for the next move in the current game"""
    try:
        result = games_module.get_hint(session_id=request.session_id)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to generate hint"))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating hint: {str(e)}")

# End a game
@games_router.post("/end")
async def end_game(request: SessionRequest):
    """End a game session"""
    result = games_module.end_game(session_id=request.session_id)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to end game"))
    
    return result

# Get game history
@games_router.get("/history")
async def get_game_history():
    """Get the history of played games"""
    return games_module.get_game_history()

# Get available games
@games_router.get("/available")
async def get_available_games():
    """Get list of available games and options"""
    return {
        "success": True,
        "games": [
            {
                "id": "chess",
                "name": "Chess",
                "players": 2,
                "description": "Traditional Chess game with full rules including castling, en passant, and promotion.",
                "difficulties": ["easy", "medium", "hard", "expert"],
                "themes": ["default", "classic", "modern", "wood", "tournament"]
            },
            {
                "id": "go",
                "name": "Go",
                "players": 2,
                "description": "Traditional Go (Weiqi/Baduk) with customizable board sizes and rule sets.",
                "board_sizes": [9, 13, 19],
                "difficulties": ["easy", "medium", "hard", "expert"],
                "themes": ["default", "classic", "wooden", "modern"]
            },
            {
                "id": "mahjong",
                "name": "Mahjong",
                "players": 4,
                "description": "Traditional Riichi Mahjong with complete rule set including calls and scoring.",
                "difficulties": ["easy", "medium", "hard", "expert"],
                "themes": ["default", "traditional", "simple"]
            }
        ]
    }

# Function to include the games router in the main Sully API
def include_games_router(app):
    """Add the games router to the main FastAPI app"""
    app.include_router(games_router)