import datetime 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class Domain(BaseModel):
    id: int
    domain: str
    full_url: Optional[str] = None
    createdat: datetime.date
    updatedat: datetime.date

# In-memory data store
domains = {}

@app.post("/domains/", response_model=Domain)
async def create_item(domain: Domain):
    if domain.id in domains:
        raise HTTPException(status_code=400, detail="Domain already exists")
    domains[domain.id] = domain
    return domain

@app.get("/domains/", response_model=List[Domain])
async def read_items():
    return list(domains.values())

# @app.get("/items/{item_id}", response_model=Item)
# async def read_item(item_id: int):
#     if item_id not in items:
#         raise HTTPException(status_code=404, detail="Item not found")
#     return items[item_id]

# @app.put("/items/{item_id}", response_model=Item)
# async def update_item(item_id: int, item: Item):
#     if item_id not in items:
#         raise HTTPException(status_code=404, detail="Item not found")
#     items[item_id] = item
#     return item

# @app.delete("/items/{item_id}")
# async def delete_item(item_id: int):
#     if item_id not in items:
#         raise HTTPException(status_code=404, detail="Item not found")
#     del items[item_id]
#     return {"detail": "Item deleted"}