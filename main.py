from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from typing import List
import uvicorn

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

class RankRequest(BaseModel):
    user_list: List[str]
    candidate_lists: List[List[str]]

@app.post("/rank")
async def rank_lists(data: RankRequest):
    def semantic_score(s1, s2):
        e1 = model.encode(s1, convert_to_tensor=True)
        e2 = model.encode(s2, convert_to_tensor=True)
        return util.cos_sim(e1, e2).item()

    def average_score(list1, list2):
        if not list1:
            return 0
        total = 0
        for s1 in list1:
            if not s1: continue
            best = max(semantic_score(s1, s2) for s2 in list2)
            total += best
        return total / len([s for s in list1 if s])

    user_list = data.user_list
    candidate_lists = data.candidate_lists

    scored = []
    for idx, candidate in enumerate(candidate_lists):
        score = average_score(user_list, candidate)
        scored.append((idx, score))

    ranked_indices = [idx for idx, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
    return {"ranked_indices": ranked_indices}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
