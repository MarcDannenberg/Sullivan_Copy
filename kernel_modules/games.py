"""
Games Kernel - A framework for implementing traditional board and tile games
"""

from abc import ABC, abstractmethod
import random
from typing import List, Dict, Any, Optional, Tuple, Set
import copy

class Player:
    """Base class for all game players"""
    def __init__(self, name: str):
        self.name = name
        self.score = 0
    
    def __str__(self):
        return f"Player: {self.name} (Score: {self.score})"

class Game(ABC):
    """Abstract base class for all games"""
    
    def __init__(self, name: str, players: List[Player]):
        self.name = name
        self.players = players
        self.current_player_idx = 0
        self.is_game_over = False
        self.winner = None
    
    @property
    def current_player(self) -> Player:
        """Returns the player whose turn it is"""
        return self.players[self.current_player_idx]
    
    def next_player(self):
        """Advances to the next player"""
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
    
    @abstractmethod
    def initialize_game(self):
        """Set up the game state"""
        pass
    
    @abstractmethod
    def is_valid_move(self, move: Any) -> bool:
        """Check if a move is valid"""
        pass
    
    @abstractmethod
    def make_move(self, move: Any) -> bool:
        """Execute a player's move and return True if successful"""
        pass
    
    @abstractmethod
    def check_game_over(self) -> bool:
        """Check if the game has ended"""
        pass
    
    @abstractmethod
    def get_game_state(self) -> Dict[str, Any]:
        """Return a representation of the current game state"""
        pass
    
    @abstractmethod
    def display(self):
        """Display the current game state"""
        pass

class GameEngine:
    """Central engine to manage games"""
    
    def __init__(self):
        self.available_games = {}
    
    def register_game(self, game_class):
        """Register a game with the engine"""
        self.available_games[game_class.__name__] = game_class
    
    def create_game(self, game_name: str, players: List[Player]) -> Optional[Game]:
        """Create a new game instance"""
        if game_name in self.available_games:
            game = self.available_games[game_name](players)
            game.initialize_game()
            return game
        return None
    
    def get_available_games(self) -> List[str]:
        """List all available games"""
        return list(self.available_games.keys())

# Example usage:
# engine = GameEngine()
# engine.register_game(MahjongGame)
# engine.register_game(ChessGame)
# engine.register_game(GoGame)
# 
# players = [Player("Player 1"), Player("Player 2")]
# mahjong_game = engine.create_game("MahjongGame", players)
# mahjong_game.display()

"""
Traditional Mahjong Game Implementation
"""

from enum import Enum, auto
from dataclasses import dataclass
from collections import Counter

class TileType(Enum):
    """Enumeration of Mahjong tile types"""
    DOTS = "Dots"  # 筒子
    BAMBOO = "Bamboo"  # 索子
    CHARACTERS = "Characters"  # 萬子
    WIND = "Wind"  # 風牌
    DRAGON = "Dragon"  # 三元牌
    FLOWER = "Flower"  # 花牌
    SEASON = "Season"  # 季節牌

class Wind(Enum):
    """Enumeration of wind directions"""
    EAST = "East"
    SOUTH = "South"
    WEST = "West"
    NORTH = "North"

class Dragon(Enum):
    """Enumeration of dragon types"""
    RED = "Red"  # 中
    GREEN = "Green"  # 發
    WHITE = "White"  # 白

@dataclass
class Tile:
    """Represents a single Mahjong tile"""
    type: TileType
    value: Any  # Number for suits, Wind/Dragon enum for winds/dragons
    
    def __str__(self):
        if self.type in [TileType.DOTS, TileType.BAMBOO, TileType.CHARACTERS]:
            return f"{self.value} {self.type.value}"
        elif self.type == TileType.WIND:
            return f"{self.value.value} Wind"
        elif self.type == TileType.DRAGON:
            return f"{self.value.value} Dragon"
        elif self.type == TileType.FLOWER:
            return f"Flower {self.value}"
        elif self.type == TileType.SEASON:
            return f"Season {self.value}"
        return "Unknown Tile"
    
    def __eq__(self, other):
        if not isinstance(other, Tile):
            return False
        return self.type == other.type and self.value == other.value
    
    def __hash__(self):
        return hash((self.type, self.value))

class MahjongHand:
    """Represents a player's hand in Mahjong"""
    
    def __init__(self):
        self.tiles = []  # Concealed tiles
        self.revealed_sets = []  # Sets revealed through calls (chi, pon, kan)
        self.discards = []  # Tiles discarded by this player
    
    def add_tile(self, tile: Tile):
        """Add a tile to the hand"""
        self.tiles.append(tile)
        # Typically in Mahjong, hands are sorted for easier viewing
        self.sort_hand()
    
    def discard_tile(self, index: int) -> Optional[Tile]:
        """Discard a tile from the hand by index"""
        if 0 <= index < len(self.tiles):
            tile = self.tiles.pop(index)
            self.discards.append(tile)
            return tile
        return None
    
    def sort_hand(self):
        """Sort the tiles in the hand for easier viewing"""
        # First by type, then by value
        def sort_key(tile):
            type_order = {
                TileType.CHARACTERS: 0,
                TileType.DOTS: 1,
                TileType.BAMBOO: 2,
                TileType.WIND: 3,
                TileType.DRAGON: 4,
                TileType.FLOWER: 5,
                TileType.SEASON: 6
            }
            
            # For winds, define a specific order
            wind_order = {
                Wind.EAST: 0,
                Wind.SOUTH: 1,
                Wind.WEST: 2,
                Wind.NORTH: 3
            }
            
            # For dragons, define a specific order
            dragon_order = {
                Dragon.WHITE: 0,
                Dragon.GREEN: 1,
                Dragon.RED: 2
            }
            
            type_val = type_order.get(tile.type, 7)
            
            if tile.type in [TileType.DOTS, TileType.BAMBOO, TileType.CHARACTERS]:
                return (type_val, tile.value)
            elif tile.type == TileType.WIND:
                return (type_val, wind_order.get(tile.value, 4))
            elif tile.type == TileType.DRAGON:
                return (type_val, dragon_order.get(tile.value, 3))
            else:
                return (type_val, tile.value)
        
        self.tiles.sort(key=sort_key)
    
    def can_win(self, tile: Optional[Tile] = None) -> bool:
        """
        Check if the current hand can win with the given tile.
        If tile is None, check if the current hand is a winning hand.
        """
        check_tiles = self.tiles.copy()
        if tile:
            check_tiles.append(tile)
        
        # A winning hand typically has 4 sets and a pair
        # In this simplified version, we'll just check if all tiles can form valid sets
        # A full implementation would be much more complex, checking all possible combinations
        
        # First, ensure we have the right number of tiles
        if len(check_tiles) != 14:
            return False
        
        # Create a counting dictionary
        tile_count = Counter(check_tiles)
        
        # Try to find a pair and see if rest can be divided into sets of 3
        for pair_tile, count in tile_count.items():
            if count >= 2:
                # Remove the pair
                remaining = tile_count.copy()
                remaining[pair_tile] -= 2
                
                # Check if remaining tiles can be grouped into sets
                while sum(remaining.values()) > 0:
                    found_set = False
                    
                    # Try to find a triplet
                    for t, c in remaining.items():
                        if c >= 3:
                            remaining[t] -= 3
                            found_set = True
                            break
                    
                    if not found_set:
                        # Try to find a sequence (for suited tiles only)
                        for t in remaining:
                            if t.type in [TileType.DOTS, TileType.BAMBOO, TileType.CHARACTERS] and remaining[t] > 0:
                                next_tile1 = Tile(t.type, t.value + 1)
                                next_tile2 = Tile(t.type, t.value + 2)
                                
                                if (next_tile1 in remaining and remaining[next_tile1] > 0 and 
                                    next_tile2 in remaining and remaining[next_tile2] > 0):
                                    remaining[t] -= 1
                                    remaining[next_tile1] -= 1
                                    remaining[next_tile2] -= 1
                                    found_set = True
                                    break
                    
                    if not found_set:
                        break
                
                if sum(remaining.values()) == 0:
                    return True
        
        return False
    
    def can_chi(self, tile: Tile) -> bool:
        """Check if the player can call chi (sequence) with the given tile"""
        # Chi is only valid for suited tiles (not honors) and from the player to the left
        if tile.type not in [TileType.DOTS, TileType.BAMBOO, TileType.CHARACTERS]:
            return False
        
        # Check for possible sequences
        if (Tile(tile.type, tile.value + 1) in self.tiles and 
            Tile(tile.type, tile.value + 2) in self.tiles):
            return True
        
        if (Tile(tile.type, tile.value - 1) in self.tiles and 
            Tile(tile.type, tile.value + 1) in self.tiles):
            return True
        
        if (Tile(tile.type, tile.value - 2) in self.tiles and 
            Tile(tile.type, tile.value - 1) in self.tiles):
            return True
        
        return False
    
    def can_pon(self, tile: Tile) -> bool:
        """Check if the player can call pon (triplet) with the given tile"""
        tile_count = Counter(self.tiles)
        return tile_count[tile] >= 2  # Need at least 2 matching tiles in hand
    
    def can_kan(self, tile: Tile, is_closed: bool = False) -> bool:
        """Check if the player can call kan (quad) with the given tile"""
        if is_closed:
            # For closed kan, need all 4 tiles in hand
            tile_count = Counter(self.tiles)
            return tile_count[tile] >= 4
        else:
            # For open kan, need 3 matching tiles in hand
            tile_count = Counter(self.tiles)
            return tile_count[tile] >= 3
    
    def __str__(self):
        hand_str = "Hand: " + ", ".join(str(tile) for tile in self.tiles)
        if self.revealed_sets:
            revealed_str = "Revealed: " + ", ".join(str(s) for s in self.revealed_sets)
            hand_str += "\n" + revealed_str
        return hand_str

class MahjongPlayer(Player):
    """Player in a Mahjong game"""
    
    def __init__(self, name: str, wind: Wind):
        super().__init__(name)
        self.hand = MahjongHand()
        self.wind = wind  # Player's seat wind
    
    def __str__(self):
        return f"{self.name} ({self.wind.value}) - {self.hand}"

class MahjongGame(Game):
    """Implementation of traditional Mahjong"""
    
    def __init__(self, players: List[Player]):
        if len(players) != 4:
            raise ValueError("Mahjong requires exactly 4 players")
        
        # Assign winds to players
        winds = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]
        for i, player in enumerate(players):
            if not isinstance(player, MahjongPlayer):
                # Convert regular Player to MahjongPlayer
                players[i] = MahjongPlayer(player.name, winds[i])
        
        super().__init__("Mahjong", players)
        self.wall = []  # The tile wall
        self.dead_wall = []  # Tiles for replacement draws (dora indicators, etc.)
        self.discards = []  # Discarded tiles
        self.dora_indicators = []  # Tiles that indicate the dora
        self.round_wind = Wind.EAST  # Current round wind
        self.current_discard = None  # The current tile in the discard pool (for calls)
        self.last_action = None  # Last action performed
    
    def initialize_game(self):
        """Set up the game state"""
        self.is_game_over = False
        self.winner = None
        self.wall = self._create_tile_set()
        random.shuffle(self.wall)
        
        # Separate the dead wall (14 tiles from the end)
        self.dead_wall = self.wall[-14:]
        self.wall = self.wall[:-14]
        
        # Set the first dora indicator
        self.dora_indicators = [self.dead_wall[4]]
        
        # Deal tiles to players (13 tiles each)
        for player in self.players:
            player.hand = MahjongHand()
            for _ in range(13):
                player.hand.add_tile(self.draw_tile())
        
        # East player (dealer) draws an extra tile to start
        self.current_player_idx = 0  # East player starts
        first_player = self.players[self.current_player_idx]
        first_player.hand.add_tile(self.draw_tile())
        
        # Now East player needs to discard
        self.last_action = "initialize"
    
    def _create_tile_set(self) -> List[Tile]:
        """Create a complete set of Mahjong tiles"""
        tiles = []
        
        # Suited tiles (1-9 in three suits, four of each)
        for suit in [TileType.DOTS, TileType.BAMBOO, TileType.CHARACTERS]:
            for value in range(1, 10):
                for _ in range(4):
                    tiles.append(Tile(suit, value))
        
        # Winds (4 of each)
        for wind in Wind:
            for _ in range(4):
                tiles.append(Tile(TileType.WIND, wind))
        
        # Dragons (4 of each)
        for dragon in Dragon:
            for _ in range(4):
                tiles.append(Tile(TileType.DRAGON, dragon))
        
        # Flowers and Seasons (1 of each, numbered 1-4)
        # Note: These are optional and not used in some variants
        for i in range(1, 5):
            tiles.append(Tile(TileType.FLOWER, i))
            tiles.append(Tile(TileType.SEASON, i))
        
        return tiles
    
    def draw_tile(self) -> Optional[Tile]:
        """Draw a tile from the wall"""
        if not self.wall:
            return None  # Wall is empty (exhaustive draw)
        return self.wall.pop(0)
    
    def is_valid_move(self, move: Dict[str, Any]) -> bool:
        """Check if a move is valid"""
        move_type = move.get("type")
        player_idx = move.get("player_idx", self.current_player_idx)
        player = self.players[player_idx]
        
        if move_type == "discard":
            # Check if it's this player's turn and they have the tile
            tile_idx = move.get("tile_idx")
            return player_idx == self.current_player_idx and 0 <= tile_idx < len(player.hand.tiles)
        
        elif move_type == "chi":
            # Check if the player can call chi on the current discard
            # Must be the next player's turn
            return (self.current_discard and 
                   player_idx == (self.current_player_idx + 1) % 4 and
                   player.hand.can_chi(self.current_discard))
        
        elif move_type == "pon":
            # Any player can pon if they have two matching tiles
            return (self.current_discard and 
                   player.hand.can_pon(self.current_discard))
        
        elif move_type == "kan":
            # Open kan from discard or closed kan from hand
            is_closed = move.get("is_closed", False)
            if is_closed:
                # Closed kan must be on player's turn
                tile = player.hand.tiles[move.get("tile_idx")]
                return player_idx == self.current_player_idx and player.hand.can_kan(tile, True)
            else:
                # Open kan from discard
                return self.current_discard and player.hand.can_kan(self.current_discard)
        
        elif move_type == "win":
            # Check if the player can win with the current discard or self-draw
            tile = self.current_discard if move.get("from_discard") else None
            return player.hand.can_win(tile)
        
        return False
    
    def make_move(self, move: Dict[str, Any]) -> bool:
        """Execute a player's move"""
        if not self.is_valid_move(move):
            return False
        
        move_type = move.get("type")
        player_idx = move.get("player_idx", self.current_player_idx)
        player = self.players[player_idx]
        
        if move_type == "discard":
            # Discard a tile from hand
            tile_idx = move.get("tile_idx")
            discarded_tile = player.hand.discard_tile(tile_idx)
            if discarded_tile:
                self.current_discard = discarded_tile
                self.discards.append(discarded_tile)
                self.last_action = "discard"
                # Move to next player
                self.next_player()
                # Next player draws a tile
                next_player = self.players[self.current_player_idx]
                drawn_tile = self.draw_tile()
                if drawn_tile:
                    next_player.hand.add_tile(drawn_tile)
                else:
                    # Wall is exhausted - draw game
                    self.is_game_over = True
                return True
        
        elif move_type == "chi":
            # Claim a sequence with the current discard
            sequence_tiles = move.get("sequence_tiles", [])  # Indices of tiles in hand
            if len(sequence_tiles) == 2:
                tiles = [player.hand.tiles[i] for i in sequence_tiles]
                # Form the sequence
                sequence = [self.current_discard] + tiles
                # Remove the tiles from hand
                for i in sorted(sequence_tiles, reverse=True):
                    player.hand.tiles.pop(i)
                # Add the revealed set
                player.hand.revealed_sets.append(("chi", sequence))
                self.current_discard = None
                # Set this player as current
                self.current_player_idx = player_idx
                self.last_action = "chi"
                return True
        
        elif move_type == "pon":
            # Claim a triplet with the current discard
            # Find two matching tiles in hand
            matching_tiles = [i for i, t in enumerate(player.hand.tiles) if t == self.current_discard]
            if len(matching_tiles) >= 2:
                # Remove the tiles from hand
                tiles = [player.hand.tiles[matching_tiles[0]], player.hand.tiles[matching_tiles[1]]]
                player.hand.tiles.pop(matching_tiles[1])
                player.hand.tiles.pop(matching_tiles[0])
                # Add the revealed set
                triplet = [self.current_discard] + tiles
                player.hand.revealed_sets.append(("pon", triplet))
                self.current_discard = None
                # Set this player as current
                self.current_player_idx = player_idx
                self.last_action = "pon"
                return True
        
        elif move_type == "kan":
            is_closed = move.get("is_closed", False)
            if is_closed:
                # Closed kan from hand
                tile_idx = move.get("tile_idx")
                if tile_idx < len(player.hand.tiles):
                    tile = player.hand.tiles[tile_idx]
                    # Find all four of this tile in hand
                    matching_indices = [i for i, t in enumerate(player.hand.tiles) if t == tile]
                    if len(matching_indices) == 4:
                        # Remove the tiles from hand
                        kan_tiles = []
                        for i in sorted(matching_indices, reverse=True):
                            kan_tiles.append(player.hand.tiles.pop(i))
                        # Add the revealed set
                        player.hand.revealed_sets.append(("closed_kan", kan_tiles))
                        # Draw replacement tile from dead wall
                        if self.dead_wall:
                            player.hand.add_tile(self.dead_wall.pop(0))
                        self.last_action = "closed_kan"
                        # Add new dora indicator
                        if len(self.dora_indicators) < 5 and self.dead_wall:
                            self.dora_indicators.append(self.dead_wall[4 + len(self.dora_indicators)])
                        return True
            else:
                # Open kan from discard
                matching_indices = [i for i, t in enumerate(player.hand.tiles) if t == self.current_discard]
                if len(matching_indices) >= 3:
                    # Remove the tiles from hand
                    kan_tiles = []
                    for i in sorted(matching_indices[:3], reverse=True):
                        kan_tiles.append(player.hand.tiles.pop(i))
                    # Add the current discard
                    kan_tiles.append(self.current_discard)
                    # Add the revealed set
                    player.hand.revealed_sets.append(("open_kan", kan_tiles))
                    self.current_discard = None
                    # Set this player as current
                    self.current_player_idx = player_idx
                    # Draw replacement tile from dead wall
                    if self.dead_wall:
                        player.hand.add_tile(self.dead_wall.pop(0))
                    self.last_action = "open_kan"
                    # Add new dora indicator
                    if len(self.dora_indicators) < 5 and self.dead_wall:
                        self.dora_indicators.append(self.dead_wall[4 + len(self.dora_indicators)])
                    return True
        
        elif move_type == "win":
            # Player wins the game
            from_discard = move.get("from_discard", False)
            win_tile = self.current_discard if from_discard else None
            if player.hand.can_win(win_tile):
                self.winner = player
                self.is_game_over = True
                self.last_action = "win"
                # In a real game, we would calculate scores here
                return True
        
        return False
    
    def check_game_over(self) -> bool:
        """Check if the game has ended"""
        # Game can end by someone winning or exhaustive draw
        return self.is_game_over
    
    def get_game_state(self) -> Dict[str, Any]:
        """Return a representation of the current game state"""
        return {
            "current_player": self.current_player_idx,
            "round_wind": self.round_wind.value,
            "dora_indicators": [str(tile) for tile in self.dora_indicators],
            "wall_size": len(self.wall),
            "dead_wall_size": len(self.dead_wall),
            "discards": [str(tile) for tile in self.discards],
            "current_discard": str(self.current_discard) if self.current_discard else None,
            "players": [{
                "name": player.name,
                "wind": player.wind.value,
                "hand": [str(tile) for tile in player.hand.tiles],
                "revealed_sets": player.hand.revealed_sets,
                "discards": [str(tile) for tile in player.hand.discards]
            } for player in self.players],
            "is_game_over": self.is_game_over,
            "winner": self.winner.name if self.winner else None,
            "last_action": self.last_action
        }
    
    def display(self):
        """Display the current game state"""
        print(f"=== Mahjong Game ===")
        print(f"Round Wind: {self.round_wind.value}")
        print(f"Dora Indicator(s): {', '.join(str(tile) for tile in self.dora_indicators)}")
        print(f"Wall: {len(self.wall)} tiles remaining")
        print(f"Current Discard: {self.current_discard}")
        print("\nPlayers:")
        for i, player in enumerate(self.players):
            current = "→ " if i == self.current_player_idx else "  "
            print(f"{current}{player}")
        
        if self.is_game_over:
            if self.winner:
                print(f"\nGame Over! Winner: {self.winner.name}")
            else:
                print("\nGame Over! Exhaustive Draw")

# Example usage:
# winds = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]
# players = [MahjongPlayer(f"Player {i+1}", wind) for i, wind in enumerate(winds)]
# game = MahjongGame(players)
# game.initialize_game()
# game.display()

"""
Go Game Implementation
"""

class Stone(Enum):
    """Enum representing stone colors in Go"""
    BLACK = "●"
    WHITE = "○"
    EMPTY = "·"

class GoPlayer(Player):
    """Player in a Go game"""
    
    def __init__(self, name: str, stone: Stone):
        super().__init__(name)
        self.stone = stone
        self.captures = 0
    
    def __str__(self):
        return f"{self.name} ({self.stone.value}, Captures: {self.captures})"

class GoGame(Game):
    """Implementation of the Go board game"""
    
    def __init__(self, players: List[Player], board_size: int = 19):
        if len(players) != 2:
            raise ValueError("Go requires exactly 2 players")
        
        # Convert regular Players to GoPlayers if needed
        stones = [Stone.BLACK, Stone.WHITE]
        for i, player in enumerate(players):
            if not isinstance(player, GoPlayer):
                players[i] = GoPlayer(player.name, stones[i])
        
        super().__init__("Go", players)
        self.board_size = board_size
        self.board = None
        self.previous_board = None  # For ko rule checking
        self.consecutive_passes = 0
        self.move_history = []
    
    def initialize_game(self):
        """Set up the game state"""
        self.is_game_over = False
        self.winner = None
        # Create empty board
        self.board = [[Stone.EMPTY for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.previous_board = copy.deepcopy(self.board)
        self.consecutive_passes = 0
        self.move_history = []
        # Black plays first in Go
        self.current_player_idx = 0
    
    def is_valid_move(self, move: Dict[str, Any]) -> bool:
        """Check if a move is valid"""
        move_type = move.get("type")
        
        if move_type == "pass":
            # Passing is always valid
            return True
        
        elif move_type == "place":
            x, y = move.get("x"), move.get("y")
            
            # Check if coordinates are valid
            if not (0 <= x < self.board_size and 0 <= y < self.board_size):
                return False
            
            # Check if the intersection is empty
            if self.board[y][x] != Stone.EMPTY:
                return False
            
            # Create a hypothetical board for testing
            test_board = copy.deepcopy(self.board)
            test_board[y][x] = self.current_player.stone
            
            # Check if the move would capture any stones
            captured = self._find_captures(test_board, x, y)
            
            # If no captures, check if the move would result in a suicide
            if not captured:
                # Check if the placed stone has liberties
                if not self._has_liberties(test_board, x, y):
                    return False
            
            # Check for ko rule - simple ko check
            # If the move would exactly recreate the previous board position, it's invalid
            if self._would_violate_ko(test_board, captured, x, y):
                return False
            
            return True
        
        return False
    
    def _would_violate_ko(self, test_board: List[List[Stone]], captured: List[Tuple[int, int]], x: int, y: int) -> bool:
        """Check if a move would violate the ko rule"""
        # If there's exactly one capture and that capture would recreate the previous board position
        if len(captured) == 1:
            # Apply captures to test board
            for cx, cy in captured:
                test_board[cy][cx] = Stone.EMPTY
            
            # Check if the resulting board is identical to the previous one
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if test_board[i][j] != self.previous_board[i][j]:
                        return False
            
            return True
        
        return False
    
    def _has_liberties(self, board: List[List[Stone]], x: int, y: int) -> bool:
        """Check if a stone or group has liberties (empty adjacent intersections)"""
        stone = board[y][x]
        if stone == Stone.EMPTY:
            return True
        
        checked = set()
        return self._check_liberties(board, x, y, stone, checked)
    
    def _check_liberties(self, board: List[List[Stone]], x: int, y: int, stone: Stone, checked: Set[Tuple[int, int]]) -> bool:
        """Recursive function to check if a stone or group has liberties"""
        if (x, y) in checked:
            return False
        
        checked.add((x, y))
        
        # Check adjacent intersections
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # Check if the new coordinates are valid
            if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                continue
            
            # If there's an empty intersection, the group has a liberty
            if board[ny][nx] == Stone.EMPTY:
                return True
            
            # If there's a stone of the same color, check if it has liberties
            if board[ny][nx] == stone and self._check_liberties(board, nx, ny, stone, checked):
                return True
        
        return False
    
    def _find_captures(self, board: List[List[Stone]], x: int, y: int) -> List[Tuple[int, int]]:
        """Find any opponent stones that would be captured by placing a stone at (x, y)"""
        stone = board[y][x]
        opponent_stone = Stone.WHITE if stone == Stone.BLACK else Stone.BLACK
        captures = []
        
        # Check adjacent intersections for opponent stones
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # Check if the new coordinates are valid
            if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                continue
            
            # If there's an opponent stone, check if it/its group has liberties
            if board[ny][nx] == opponent_stone:
                group = self._find_group(board, nx, ny)
                has_liberty = False
                
                for gx, gy in group:
                    # Check adjacent intersections for liberties
                    for gdx, gdy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        gnx, gny = gx + gdx, gy + gdy
                        
                        # Check if the new coordinates are valid
                        if not (0 <= gnx < self.board_size and 0 <= gny < self.board_size):
                            continue
                        
                        # If there's an empty intersection, the group has a liberty
                        if board[gny][gnx] == Stone.EMPTY:
                            has_liberty = True
                            break
                    
                    if has_liberty:
                        break
                
                # If the group has no liberties, it would be captured
                if not has_liberty:
                    captures.extend(group)
        
        return captures
    
    def _find_group(self, board: List[List[Stone]], x: int, y: int) -> List[Tuple[int, int]]:
        """Find all stones in a connected group"""
        stone = board[y][x]
        if stone == Stone.EMPTY:
            return []
        
        group = []
        checked = set()
        self._collect_group(board, x, y, stone, group, checked)
        return group
    
    def _collect_group(self, board: List[List[Stone]], x: int, y: int, stone: Stone, 
                      group: List[Tuple[int, int]], checked: Set[Tuple[int, int]]):
        """Recursive function to collect all stones in a connected group"""
        if (x, y) in checked:
            return
        
        checked.add((x, y))
        
        if board[y][x] == stone:
            group.append((x, y))
            
            # Check adjacent intersections
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                # Check if the new coordinates are valid
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    self._collect_group(board, nx, ny, stone, group, checked)
    
    def make_move(self, move: Dict[str, Any]) -> bool:
        """Execute a player's move"""
        if not self.is_valid_move(move):
            return False
        
        move_type = move.get("type")
        
        if move_type == "pass":
            # Record the pass
            self.move_history.append({"player": self.current_player_idx, "type": "pass"})
            self.consecutive_passes += 1
            
            # Check if both players passed consecutively
            if self.consecutive_passes >= 2:
                self.is_game_over = True
                # Score the game
                self._score_game()
            else:
                # Move to the next player
                self.next_player()
            
            return True
        
        elif move_type == "place":
            x, y = move.get("x"), move.get("y")
            
            # Save the current board for ko rule checking
            self.previous_board = copy.deepcopy(self.board)
            
            # Place the stone
            self.board[y][x] = self.current_player.stone
            
            # Find and remove any captured stones
            captures = self._find_captures(self.board, x, y)
            for cx, cy in captures:
                self.board[cy][cx] = Stone.EMPTY
            
            # Update the player's capture count
            self.current_player.captures += len(captures)
            
            # Record the move
            self.move_history.append({
                "player": self.current_player_idx,
                "type": "place",
                "x": x,
                "y": y,
                "captures": len(captures)
            })
            
            # Reset consecutive passes
            self.consecutive_passes = 0
            
            # Move to the next player
            self.next_player()
            
            return True
        
        return False
    
    def _score_game(self):
        """Score the game at the end"""
        # Chinese scoring: territory + stones on the board
        black_score = 0
        white_score = 0
        
        # Count stones on the board
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] == Stone.BLACK:
                    black_score += 1
                elif self.board[y][x] == Stone.WHITE:
                    white_score += 1
        
        # Add komi (compensation for black's first move advantage)
        white_score += 6.5  # Standard komi
        
        # Determine the winner
        if black_score > white_score:
            self.winner = self.players[0]  # Black
        else:
            self.winner = self.players[1]  # White
        
        # Set final scores
        self.players[0].score = black_score
        self.players[1].score = white_score
    
    def check_game_over(self) -> bool:
        """Check if the game has ended"""
        return self.is_game_over
    
    def get_game_state(self) -> Dict[str, Any]:
        """Return a representation of the current game state"""
        return {
            "current_player": self.current_player_idx,
            "board": [[stone.value for stone in row] for row in self.board],
            "consecutive_passes": self.consecutive_passes,
            "move_history": self.move_history,
            "black_captures": self.players[0].captures,
            "white_captures": self.players[1].captures,
            "is_game_over": self.is_game_over,
            "winner": self.winner.name if self.winner else None,
            "black_score": self.players[0].score,
            "white_score": self.players[1].score
        }
    
    def display(self):
        """Display the current game state"""
        print(f"=== Go Game ({self.board_size}x{self.board_size}) ===")
        print(f"Current Player: {self.current_player.name} ({self.current_player.stone.value})")
        
        # Display the board
        board_str = "  "
        for i in range(self.board_size):
            board_str += f"{i:2d}"
        board_str += "\n"
        
        for y in range(self.board_size):
            board_str += f"{y:2d}"
            for x in range(self.board_size):
                board_str += f" {self.board[y][x].value}"
            board_str += "\n"
        
        print(board_str)
        
        print(f"Black Captures: {self.players[0].captures}")
        print(f"White Captures: {self.players[1].captures}")
        
        if self.is_game_over:
            print(f"\nGame Over!")
            print(f"Black Score: {self.players[0].score}")
            print(f"White Score: {self.players[1].score}")
            print(f"Winner: {self.winner.name}")

# Example usage:
# players = [Player("Black Player"), Player("White Player")]
# game = GoGame(players, board_size=19)
# game.initialize_game()
# game.display()
# 
# # Making a move
# move = {"type": "place", "x": 3, "y": 3}
# if game.is_valid_move(move):
#     game.make_move(move)
#     game.display()

"""
Chess Game Implementation
"""

class PieceType(Enum):
    """Enum representing chess piece types"""
    PAWN = "P"
    KNIGHT = "N"
    BISHOP = "B"
    ROOK = "R"
    QUEEN = "Q"
    KING = "K"
    EMPTY = " "

class PieceColor(Enum):
    """Enum representing chess piece colors"""
    WHITE = "W"
    BLACK = "B"
    NONE = "N"

class ChessPiece:
    """Represents a chess piece"""
    
    def __init__(self, piece_type: PieceType, color: PieceColor):
        self.type = piece_type
        self.color = color
        self.has_moved = False  # Useful for castling and initial pawn movement
    
    def __str__(self):
        if self.type == PieceType.EMPTY:
            return " "
        symbol = self.type.value
        return symbol.upper() if self.color == PieceColor.WHITE else symbol.lower()
    
    def __eq__(self, other):
        if not isinstance(other, ChessPiece):
            return False
        return self.type == other.type and self.color == other.color

class ChessPlayer(Player):
    """Player in a chess game"""
    
    def __init__(self, name: str, color: PieceColor):
        super().__init__(name)
        self.color = color
        self.captured_pieces = []
    
    def __str__(self):
        captured = ", ".join(str(p) for p in self.captured_pieces) if self.captured_pieces else "None"
        return f"{self.name} ({self.color.name}, Captured: {captured})"

class ChessGame(Game):
    """Implementation of the chess game"""
    
    def __init__(self, players: List[Player]):
        if len(players) != 2:
            raise ValueError("Chess requires exactly 2 players")
        
        # Assign colors to players
        colors = [PieceColor.WHITE, PieceColor.BLACK]
        for i, player in enumerate(players):
            if not isinstance(player, ChessPlayer):
                players[i] = ChessPlayer(player.name, colors[i])
        
        super().__init__("Chess", players)
        self.board = None
        self.move_history = []
        self.check_status = False
        self.checkmate_status = False
        self.stalemate_status = False
        self.en_passant_target = None  # Coordinates for potential en passant capture
    
    def initialize_game(self):
        """Set up the game state"""
        self.is_game_over = False
        self.winner = None
        self.check_status = False
        self.checkmate_status = False
        self.stalemate_status = False
        self.en_passant_target = None
        self.move_history = []
        
        # Create empty board (8x8)
        self.board = [[ChessPiece(PieceType.EMPTY, PieceColor.NONE) for _ in range(8)] for _ in range(8)]
        
        # Set up pawns
        for col in range(8):
            self.board[1][col] = ChessPiece(PieceType.PAWN, PieceColor.WHITE)
            self.board[6][col] = ChessPiece(PieceType.PAWN, PieceColor.BLACK)
        
        # Set up other pieces
        # White pieces
        self.board[0][0] = ChessPiece(PieceType.ROOK, PieceColor.WHITE)
        self.board[0][1] = ChessPiece(PieceType.KNIGHT, PieceColor.WHITE)
        self.board[0][2] = ChessPiece(PieceType.BISHOP, PieceColor.WHITE)
        self.board[0][3] = ChessPiece(PieceType.QUEEN, PieceColor.WHITE)
        self.board[0][4] = ChessPiece(PieceType.KING, PieceColor.WHITE)
        self.board[0][5] = ChessPiece(PieceType.BISHOP, PieceColor.WHITE)
        self.board[0][6] = ChessPiece(PieceType.KNIGHT, PieceColor.WHITE)
        self.board[0][7] = ChessPiece(PieceType.ROOK, PieceColor.WHITE)
        
        # Black pieces
        self.board[7][0] = ChessPiece(PieceType.ROOK, PieceColor.BLACK)
        self.board[7][1] = ChessPiece(PieceType.KNIGHT, PieceColor.BLACK)
        self.board[7][2] = ChessPiece(PieceType.BISHOP, PieceColor.BLACK)
        self.board[7][3] = ChessPiece(PieceType.QUEEN, PieceColor.BLACK)
        self.board[7][4] = ChessPiece(PieceType.KING, PieceColor.BLACK)
        self.board[7][5] = ChessPiece(PieceType.BISHOP, PieceColor.BLACK)
        self.board[7][6] = ChessPiece(PieceType.KNIGHT, PieceColor.BLACK)
        self.board[7][7] = ChessPiece(PieceType.ROOK, PieceColor.BLACK)
        
        # White starts in chess
        self.current_player_idx = 0
    
    def is_valid_move(self, move: Dict[str, Any]) -> bool:
        """Check if a move is valid"""
        from_x, from_y = move.get("from_x"), move.get("from_y")
        to_x, to_y = move.get("to_x"), move.get("to_y")
        
        # Check if coordinates are valid
        if not (0 <= from_x < 8 and 0 <= from_y < 8 and 0 <= to_x < 8 and 0 <= to_y < 8):
            return False
        
        # Get the piece to move
        piece = self.board[from_y][from_x]
        
        # Check if the piece belongs to the current player
        if piece.color != self.current_player.color:
            return False
        
        # Check if the destination is empty or contains an opponent's piece
        dest_piece = self.board[to_y][to_x]
        if dest_piece.color == piece.color:
            return False
        
        # Check if the move is valid for the specific piece type
        if not self._is_piece_move_valid(piece, from_x, from_y, to_x, to_y):
            return False
        
        # Check for special moves
        promotion = move.get("promotion")
        if promotion and not self._is_promotion_valid(piece, to_y, promotion):
            return False
        
        castling = move.get("castling", False)
        if castling and not self._is_castling_valid(piece, from_x, from_y, to_x, to_y):
            return False
        
        en_passant = move.get("en_passant", False)
        if en_passant and not self._is_en_passant_valid(piece, from_x, from_y, to_x, to_y):
            return False
        
        # Check if the move would put/leave the player in check
        if self._would_be_in_check(from_x, from_y, to_x, to_y):
            return False
        
        return True
    
    def _is_piece_move_valid(self, piece: ChessPiece, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Check if a move is valid for the specific piece type"""
        # Calculate deltas
        dx = to_x - from_x
        dy = to_y - from_y
        
        if piece.type == PieceType.PAWN:
            # Pawns move differently depending on color
            direction = 1 if piece.color == PieceColor.WHITE else -1
            
            # Normal forward movement (one square)
            if dx == 0 and dy == direction and self.board[to_y][to_x].type == PieceType.EMPTY:
                return True
            
            # Initial two-square movement
            if (dx == 0 and dy == 2 * direction and 
                not piece.has_moved and 
                self.board[from_y + direction][from_x].type == PieceType.EMPTY and 
                self.board[to_y][to_x].type == PieceType.EMPTY):
                return True
            
            # Capturing diagonally
            if abs(dx) == 1 and dy == direction:
                # Regular capture
                if self.board[to_y][to_x].type != PieceType.EMPTY:
                    return True
                
                # En passant capture
                if self.en_passant_target == (to_x, to_y):
                    return True
            
            return False
        
        elif piece.type == PieceType.KNIGHT:
            # Knights move in an L-shape
            return (abs(dx) == 2 and abs(dy) == 1) or (abs(dx) == 1 and abs(dy) == 2)
        
        elif piece.type == PieceType.BISHOP:
            # Bishops move diagonally
            if abs(dx) != abs(dy):
                return False
            
            # Check if the path is clear
            return self._is_path_clear(from_x, from_y, to_x, to_y)
        
        elif piece.type == PieceType.ROOK:
            # Rooks move horizontally or vertically
            if dx != 0 and dy != 0:
                return False
            
            # Check if the path is clear
            return self._is_path_clear(from_x, from_y, to_x, to_y)
        
        elif piece.type == PieceType.QUEEN:
            # Queens move like a rook or bishop
            if dx != 0 and dy != 0 and abs(dx) != abs(dy):
                return False
            
            # Check if the path is clear
            return self._is_path_clear(from_x, from_y, to_x, to_y)
        
        elif piece.type == PieceType.KING:
            # Kings move one square in any direction
            if abs(dx) <= 1 and abs(dy) <= 1:
                return True
            
            # Castling (handled separately in _is_castling_valid)
            if abs(dx) == 2 and dy == 0:
                return self._is_castling_valid(piece, from_x, from_y, to_x, to_y)
            
            return False
        
        return False
    
    def _is_path_clear(self, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Check if the path between two positions is clear of pieces"""
        dx = to_x - from_x
        dy = to_y - from_y
        
        # Determine step direction
        x_step = 0 if dx == 0 else (1 if dx > 0 else -1)
        y_step = 0 if dy == 0 else (1 if dy > 0 else -1)
        
        # Start from the square after the origin
        x, y = from_x + x_step, from_y + y_step
        
        # Check each square along the path
        while (x, y) != (to_x, to_y):
            if self.board[y][x].type != PieceType.EMPTY:
                return False
            x += x_step
            y += y_step
        
        return True
    
    def _is_path_clear_on_board(self, from_x: int, from_y: int, to_x: int, to_y: int, board) -> bool:
        """Check if the path between two positions is clear on a given board"""
        dx = to_x - from_x
        dy = to_y - from_y
        
        # Determine step direction
        x_step = 0 if dx == 0 else (1 if dx > 0 else -1)
        y_step = 0 if dy == 0 else (1 if dy > 0 else -1)
        
        # Start from the square after the origin
        x, y = from_x + x_step, from_y + y_step
        
        # Check each square along the path
        while (x, y) != (to_x, to_y):
            if board[y][x].type != PieceType.EMPTY:
                return False
            x += x_step
            y += y_step
        
        return True
    
    def _is_promotion_valid(self, piece: ChessPiece, to_y: int, promotion: str) -> bool:
        """Check if a pawn promotion is valid"""
        # Promotion is only valid for pawns reaching the opposite edge
        if piece.type != PieceType.PAWN:
            return False
        
        # Check if the pawn is reaching the opposite edge
        if (piece.color == PieceColor.WHITE and to_y != 7) or (piece.color == PieceColor.BLACK and to_y != 0):
            return False
        
        # Check if the promotion type is valid
        valid_promotions = ["Q", "N", "R", "B"]
        return promotion.upper() in valid_promotions
    
    def _is_castling_valid(self, piece: ChessPiece, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Check if castling is valid"""
        # Castling is only valid for kings
        if piece.type != PieceType.KING:
            return False
        
        # King must not have moved
        if piece.has_moved:
            return False
        
        # King must be on the correct starting position
        if from_y != 0 and from_y != 7:
            return False
        if from_x != 4:
            return False
        
        # Must be a horizontal move of two squares
        if from_y != to_y or abs(to_x - from_x) != 2:
            return False
        
        # Determine rook position
        rook_x = 0 if to_x < from_x else 7  # Queenside or kingside
        rook = self.board[from_y][rook_x]
        
        # Check if there's a rook in the correct position
        if rook.type != PieceType.ROOK or rook.color != piece.color:
            return False
        
        # Rook must not have moved
        if rook.has_moved:
            return False
        
        # Path between king and rook must be clear
        min_x = min(from_x, rook_x) + 1
        max_x = max(from_x, rook_x)
        for x in range(min_x, max_x):
            if self.board[from_y][x].type != PieceType.EMPTY:
                return False
        
        # King must not be in check
        if self._is_in_check(piece.color):
            return False
        
        # King must not pass through or land on a square under attack
        step = -1 if to_x < from_x else 1
        for x in range(from_x, to_x + step, step):
            if self._is_square_attacked(x, from_y, piece.color):
                return False
        
        return True
    
    def _is_en_passant_valid(self, piece: ChessPiece, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Check if an en passant capture is valid"""
        # En passant is only valid for pawns
        if piece.type != PieceType.PAWN:
            return False
        
        # Must be a diagonal move
        if abs(to_x - from_x) != 1:
            return False
        
        # Ensure the destination is the en passant target
        return self.en_passant_target == (to_x, to_y)
    
    def _would_be_in_check(self, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Check if a move would leave the player in check"""
        # Create a copy of the board
        board_copy = copy.deepcopy(self.board)
        
        # Temporarily make the move
        board_copy[to_y][to_x] = board_copy[from_y][from_x]
        board_copy[from_y][from_x] = ChessPiece(PieceType.EMPTY, PieceColor.NONE)
        
        # Find the king's position
        king_pos = None
        for y in range(8):
            for x in range(8):
                piece = board_copy[y][x]
                if (piece.type == PieceType.KING and piece.color == self.current_player.color):
                    king_pos = (x, y)
                    break
            if king_pos:
                break
        
        if not king_pos:
            return True  # If king not found, assume invalid
        
        # Check if any opponent's piece can capture the king
        for y in range(8):
            for x in range(8):
                piece = board_copy[y][x]
                if piece.type != PieceType.EMPTY and piece.color != self.current_player.color:
                    if self._can_piece_move_to(piece, x, y, king_pos[0], king_pos[1], board_copy):
                        return True
        
        return False
    
    def _can_piece_move_to(self, piece: ChessPiece, from_x: int, from_y: int, to_x: int, to_y: int, board) -> bool:
        """Check if a piece can move to a specific position on a given board"""
        # Calculate deltas
        dx = to_x - from_x
        dy = to_y - from_y
        
        if piece.type == PieceType.PAWN:
            # Pawns can only capture diagonally
            direction = 1 if piece.color == PieceColor.WHITE else -1
            return abs(dx) == 1 and dy == direction
        
        elif piece.type == PieceType.KNIGHT:
            return (abs(dx) == 2 and abs(dy) == 1) or (abs(dx) == 1 and abs(dy) == 2)
        
        elif piece.type == PieceType.BISHOP:
            if abs(dx) != abs(dy):
                return False
            
            # Check if the path is clear
            return self._is_path_clear_on_board(from_x, from_y, to_x, to_y, board)
        
        elif piece.type == PieceType.ROOK:
            if dx != 0 and dy != 0:
                return False
            
            # Check if the path is clear
            return self._is_path_clear_on_board(from_x, from_y, to_x, to_y, board)
        
        elif piece.type == PieceType.QUEEN:
            if dx != 0 and dy != 0 and abs(dx) != abs(dy):
                return False
            
            # Check if the path is clear
            return self._is_path_clear_on_board(from_x, from_y, to_x, to_y, board)
        
        elif piece.type == PieceType.KING:
            # Kings move one square in any direction
            return abs(dx) <= 1 and abs(dy) <= 1
        
        return False
    
    def _is_in_check(self, color: PieceColor) -> bool:
        """Check if a player is in check"""
        # Find the king's position
        king_pos = None
        for y in range(8):
            for x in range(8):
                piece = self.board[y][x]
                if piece.type == PieceType.KING and piece.color == color:
                    king_pos = (x, y)
                    break
            if king_pos:
                break
        
        if not king_pos:
            return False  # Should not happen in a valid game
        
        # Check if the king is under attack
        return self._is_square_attacked(king_pos[0], king_pos[1], color)
    
    def _is_square_attacked(self, x: int, y: int, color: PieceColor) -> bool:
        """Check if a square is under attack by the opponent"""
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if piece.type != PieceType.EMPTY and piece.color != color:
                    if self._can_piece_move_to(piece, j, i, x, y, self.board):
                        return True
        return False
    
    def make_move(self, move: Dict[str, Any]) -> bool:
        """Execute a player's move"""
        if not self.is_valid_move(move):
            return False
        
        from_x, from_y = move.get("from_x"), move.get("from_y")
        to_x, to_y = move.get("to_x"), move.get("to_y")
        
        # Get the piece being moved
        piece = self.board[from_y][from_x]
        
        # Handle en passant
        en_passant_capture = False
        en_passant_captured_pos = None
        if piece.type == PieceType.PAWN and self.en_passant_target == (to_x, to_y):
            en_passant_capture = True
            en_passant_captured_pos = (to_x, from_y)  # The actual position of the captured pawn
        
        # Update en passant target for next move
        self.en_passant_target = None
        if piece.type == PieceType.PAWN and abs(to_y - from_y) == 2:
            # Pawn moved two squares, set en passant target
            y_middle = (from_y + to_y) // 2
            self.en_passant_target = (from_x, y_middle)
        
        # Handle castling
        rook_move = None
        if piece.type == PieceType.KING and abs(to_x - from_x) == 2:
            # Determine rook position and new position
            rook_x = 0 if to_x < from_x else 7
            new_rook_x = 3 if to_x < from_x else 5
            rook_move = (rook_x, from_y, new_rook_x, from_y)
        
        # Move the piece
        captured_piece = None
        if self.board[to_y][to_x].type != PieceType.EMPTY:
            captured_piece = self.board[to_y][to_x]
            self.current_player.captured_pieces.append(captured_piece)
        
        self.board[to_y][to_x] = piece
        self.board[from_y][from_x] = ChessPiece(PieceType.EMPTY, PieceColor.NONE)
        
        # Handle en passant capture
        if en_passant_capture:
            captured_pawn = self.board[en_passant_captured_pos[1]][en_passant_captured_pos[0]]
            self.current_player.captured_pieces.append(captured_pawn)
            self.board[en_passant_captured_pos[1]][en_passant_captured_pos[0]] = ChessPiece(PieceType.EMPTY, PieceColor.NONE)
        
        # Execute castling rook move
        if rook_move:
            rx, ry, new_rx, new_ry = rook_move
            self.board[new_ry][new_rx] = self.board[ry][rx]
            self.board[ry][rx] = ChessPiece(PieceType.EMPTY, PieceColor.NONE)
        
        # Handle promotion
        promotion = move.get("promotion")
        if promotion and piece.type == PieceType.PAWN and (to_y == 0 or to_y == 7):
            promotion_type_map = {
                "Q": PieceType.QUEEN,
                "N": PieceType.KNIGHT,
                "R": PieceType.ROOK,
                "B": PieceType.BISHOP
            }
            promotion_type = promotion_type_map.get(promotion.upper(), PieceType.QUEEN)
            self.board[to_y][to_x] = ChessPiece(promotion_type, piece.color)
        
        # Mark the piece as moved
        piece.has_moved = True
        
        # Record the move
        move_record = {
            "piece": piece.type.value,
            "from": (from_x, from_y),
            "to": (to_x, to_y),
            "captured": captured_piece.type.value if captured_piece else None,
            "promotion": promotion
        }
        self.move_history.append(move_record)
        
        # Check for checkmate or stalemate
        self.next_player()
        
        # Check if the next player is in check
        self.check_status = self._is_in_check(self.current_player.color)
        
        # Check if the game is over
        if self.check_game_over():
            self.is_game_over = True
            # Set winner if it's checkmate
            if self.checkmate_status:
                self.winner = self.players[(self.current_player_idx + 1) % 2]
        
        return True
    
    def check_game_over(self) -> bool:
        """Check if the game has ended (checkmate or stalemate)"""
        # If the current player has no valid moves
        if not self._has_valid_moves():
            if self.check_status:
                # Checkmate
                self.checkmate_status = True
                return True
            else:
                # Stalemate
                self.stalemate_status = True
                return True
        
        # Add other end-game conditions (insufficient material, 50-move rule, etc.)
        
        return False
    
    def _has_valid_moves(self) -> bool:
        """Check if the current player has any valid moves"""
        for from_y in range(8):
            for from_x in range(8):
                piece = self.board[from_y][from_x]
                # Skip empty squares and opponent's pieces
                if piece.type == PieceType.EMPTY or piece.color != self.current_player.color:
                    continue
                
                # Check every possible destination
                for to_y in range(8):
                    for to_x in range(8):
                        # Skip the same square
                        if from_x == to_x and from_y == to_y:
                            continue
                        
                        move = {
                            "from_x": from_x,
                            "from_y": from_y,
                            "to_x": to_x,
                            "to_y": to_y
                        }
                        
                        if self.is_valid_move(move):
                            return True
        
        return False
    
    def get_game_state(self) -> Dict[str, Any]:
        """Return a representation of the current game state"""
        return {
            "current_player": self.current_player_idx,
            "board": [[str(piece) for piece in row] for row in self.board],
            "check": self.check_status,
            "checkmate": self.checkmate_status,
            "stalemate": self.stalemate_status,
            "en_passant_target": self.en_passant_target,
            "move_history": self.move_history,
            "is_game_over": self.is_game_over,
            "winner": self.winner.name if self.winner else None
        }
    
    def display(self):
        """Display the current game state"""
        print(f"=== Chess Game ===")
        print(f"Current Player: {self.current_player.name} ({self.current_player.color.name})")
        
        # Display the board
        print("  a b c d e f g h")
        print(" +-----------------+")
        for y in range(7, -1, -1):
            row_str = f"{y+1}| "
            for x in range(8):
                row_str += f"{str(self.board[y][x])} "
            row_str += f"|{y+1}"
            print(row_str)
        print(" +-----------------+")
        print("  a b c d e f g h")
        
        # Display additional game state
        if self.check_status:
            print("CHECK!")
        if self.checkmate_status:
            print("CHECKMATE!")
        if self.stalemate_status:
            print("STALEMATE!")
        
        if self.is_game_over:
            if self.winner:
                print(f"Game Over! Winner: {self.winner.name}")
            else:
                print("Game Over! Draw")

# Example usage:
# players = [Player("White Player"), Player("Black Player")]
# game = ChessGame(players)
# game.initialize_game()
# game.display()
# 
# # Making a move (e2 to e4)
# move = {"from_x": 4, "from_y": 1, "to_x": 4, "to_y": 3}
# if game.is_valid_move(move):
#     game.make_move(move)
#     game.display()

# Main entry point to demonstrate usage
if __name__ == "__main__":
    engine = GameEngine()
    
    # Register games
    engine.register_game(MahjongGame)
    engine.register_game(ChessGame)
    engine.register_game(GoGame)
    
    # List available games
    print("Available games:")
    for game_name in engine.get_available_games():
        print(f"- {game_name}")
    
    # Example: Create and initialize a chess game
    print("\nCreating Chess game...")
    players = [Player("White Player"), Player("Black Player")]
    chess_game = engine.create_game("ChessGame", players)
    chess_game.display()
    
    # Make a sample move (e2 to e4)
    print("\nMaking a move (e2 to e4)...")
    move = {"from_x": 4, "from_y": 1, "to_x": 4, "to_y": 3}
    if chess_game.is_valid_move(move):
        chess_game.make_move(move)
        chess_game.display()
    else:
        print("Invalid move!")