import sys

def main():
    if len(sys.argv) < 2:
        print("AI Recommendation Engine Prototype")
        print("Usage: python main.py [evaluate | api]")
        print("  evaluate : Run the offline Evaluation logic (Hit Rate@5, etc)")
        print("  api      : Start the FastAPI server to serve recommendations")
        sys.exit(1)
        
    command = sys.argv[1].lower()
    
    if command == "evaluate":
        from src.evaluation import evaluate_system
        evaluate_system()
    elif command == "api":
        from src.api_layer import start_server
        start_server()
    elif command == "report":
        from src.reporting import generate_reports
        generate_reports()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py [evaluate | api | report]")

if __name__ == "__main__":
    main()
