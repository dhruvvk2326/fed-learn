# simulate.py
import subprocess
import time
import os
import sys
import signal

def run_simulation(num_clients=3):
    """Run federated learning simulation with multiple clients."""
    
    # Start server in background
    print("Starting FL server...")
    server_cmd = ["python", "server.py"]
    
    if sys.platform == "win32":
        # Windows requires different settings for background processes
        server_process = subprocess.Popen(server_cmd)
    else:
        server_process = subprocess.Popen(server_cmd)
    
    # Wait for server to start
    print("Waiting for server to initialize...")
    time.sleep(5)  # Longer wait time
    
    # Start clients in separate processes
    client_processes = []
    for i in range(num_clients):
        print(f"Starting client {i}...")
        client_cmd = ["python", "client.py", f"--client-id={i}"]
        
        if sys.platform == "win32":
            proc = subprocess.Popen(client_cmd)
        else:
            proc = subprocess.Popen(client_cmd)
            
        client_processes.append(proc)
        time.sleep(1)  # Stagger client starts
    
    try:
        # Wait for server process with timeout
        print("Simulation running... (press Ctrl+C to stop)")
        server_process.wait(timeout=300)  # 5 minutes timeout
    except subprocess.TimeoutExpired:
        print("Simulation timeout reached")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        # Clean up all processes
        print("Cleaning up processes...")
        
        # Terminate clients first
        for i, proc in enumerate(client_processes):
            if proc.poll() is None:  # If process is still running
                print(f"Terminating client {i}...")
                if sys.platform == "win32":
                    proc.kill()
                else:
                    proc.terminate()
        
        # Then terminate server
        if server_process.poll() is None:
            print("Terminating server...")
            if sys.platform == "win32":
                server_process.kill()
            else:
                os.kill(server_process.pid, signal.SIGINT)  # Send SIGINT for cleaner shutdown
    
    print("Simulation completed!")

if __name__ == "__main__":
    run_simulation(num_clients=5)