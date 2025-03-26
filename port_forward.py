from pyngrok import ngrok

# Set up ngrok to expose port 4684
public_url = ngrok.connect(4684, "http")

print(f"Ngrok Tunnel URL: {public_url}")

# Keep the script running to maintain the tunnel
input("Press Enter to terminate the tunnel...")