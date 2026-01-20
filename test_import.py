import sys
sys.path.insert(0, '/workspaces/RAG-based-AI-Service')
try:
    from app.main import app
    print("✓ App imported successfully!")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {str(e)[:300]}")
