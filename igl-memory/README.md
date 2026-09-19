# IGL Memory

IGL Memory is a provenance-first memory layer for AI agents.

Core distinction:

M = immutable experience/events  
D = compiled beliefs  
Trust = whether a belief deserves confidence  
Gate = whether a trusted belief applies now  

Therefore:

Exist(D) != Trust(D) != Apply(D)

## Run

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt

uvicorn app.main:app --reload