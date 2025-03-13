from enum import Enum
from credentials import client
from pydantic import BaseModel, Field
    
class MaterialType(str, Enum):
    TABLE = "table"
    FURNACE = "furnace"
    GRASS = "grass"
    SAND = "sand"
    LAVA = "lava"
    TREE = "tree"
    WATER = "water"
    STONE = "stone"
    COAL = "coal"
    IRON = "iron"
    DIAMOND = "diamond"
    
class NavigationDestinationItems(str, Enum):
    TREE = "tree"
    WATER = "water"
    STONE = "stone"
    IRON = "iron"
    DIAMOND = "diamond"
    COAL = "coal"
    GRASS = "grass"
    COW = 'cow'
    TABLE = "table"
    FURNACE = "furnace"
    NOT_APPICABLE = "not_applicable"

class ActionType(str, Enum):
    noop = "noop"
    move_left = "move_left"
    move_right = "move_right"
    move_up = "move_up"
    move_down = "move_down"
    do = "do"
    # sleep = "sleep"
    # place_stone = "place_stone"
    # place_table = "place_table"
    # place_furnace = "place_furnace"
    # place_plant = "place_plant"
    # make_wood_pickaxe = "make_wood_pickaxe"
    # make_stone_pickaxe = "make_stone_pickaxe"
    # make_iron_pickaxe = "make_iron_pickaxe"
    # navigator = "navigator"
    # share = "share"

class InventoryItems(str, Enum):
    WOOD = "wood"
    STONE = "stone"
    COAL = "coal"
    IRON = "iron"
    DIAMOND = "diamond"
    WOOD_PICKAXE = "wood_pickaxe"
    STONE_PICKAXE = "stone_pickaxe"
    IRON_PICKAXE = "iron_pickaxe"

class InventoryItemsCount(BaseModel):
    item: InventoryItems
    count: int

class ShareableItems(str, Enum):
    WOOD = "wood"
    STONE = "stone"
    COAL = "coal"
    IRON = "iron"
    DIAMOND = "diamond"
    WOOD_PICKAXE = "wood_pickaxe"
    STONE_PICKAXE = "stone_pickaxe"
    IRON_PICKAXE = "iron_pickaxe"
    NOT_APPLICABLE = "not_applicable"

class Response(BaseModel):
    inventory: InventoryItemsCount = Field(description="Inventory items and their counts.")
    describe_situations: str = Field(description="What is the situation? What is your health? What are you doing?")
    plan: str = Field(description="What is the plan? Why?")
    action: ActionType = Field(description="What action should be taken?")
    # navigate_to: NavigationDestinationItems = Field(description="What is the target?")
    # share_with: int = Field(description="Who do you want to share with?")
    # share_item: ShareableItems = Field(description="What do you want to share?")
    communication_message: str = Field(description="What do you want to say to others?")
    
def get_completion(messages, model="gpt-4o"):
    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=Response,
    )
    return response.choices[0].message.parsed