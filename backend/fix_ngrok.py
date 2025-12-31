import os
import time

# 1. FORCE KILL existing ngrok processes to clear the "Already Online" error
print("💀 Killing old ngrok processes...")
os.system("taskkill /F /IM ngrok.exe >nul 2>&1")
time.sleep(2) # Wait for windows to release the ports

# 2. WRITE the correct configuration file
# This forces Backend to be RANDOM and Frontend to be CUSTOM
config_content = """
version: "2"
authtoken: 37ZQnyn5gkja50gAEfeItU5cfj7_44ZmikrsmSZbidKM66CRg
tunnels:
  backend_api:
    proto: http
    addr: 8000
    # No domain here -> Forces Random URL
    
  frontend_app:
    proto: http
    addr: 5173
    domain: ironic-helen-semiconsciously.ngrok-free.dev
"""

print("📝 Generating clean config file...")
with open("ngrok_clean.yml", "w") as f:
    f.write(config_content)

# 3. START ngrok using this specific file
print("🚀 Starting Ngrok... (Do not close this window)")
os.system("ngrok start --all --config=ngrok_clean.yml")