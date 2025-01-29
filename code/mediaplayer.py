import subprocess

def control_media(command, player=None):
    try:
        if player:
            if command == "play":
                subprocess.run(["playerctl", "--player", player, "play"], check=True)
            elif command == "pause":
                subprocess.run(["playerctl", "--player", player, "pause"], check=True)
            elif command == "toggle":
                subprocess.run(["playerctl", "--player", player, "play-pause"], check=True)
            elif command == "next":
                subprocess.run(["playerctl", "--player", player, "next"], check=True)
            elif command == "previous":
                subprocess.run(["playerctl", "--player", player, "previous"], check=True)
            elif command == "current":
                result = subprocess.run(
                    ["playerctl", "--player", player, "metadata", "--format", "{{artist}} - {{title}}"],
                    check=True, text=True, capture_output=True
                )
                return result.stdout.strip()
            else:
                return f"Unknown command: {command}."
        else:
            if command == "play":
                subprocess.run(["playerctl", "play"], check=True)
            elif command == "pause":
                subprocess.run(["playerctl", "pause"], check=True)
            elif command == "toggle":
                subprocess.run(["playerctl", "play-pause"], check=True)
            elif command == "next":
                subprocess.run(["playerctl", "next"], check=True)
            elif command == "previous":
                subprocess.run(["playerctl", "previous"], check=True)
            elif command == "current":
                result = subprocess.run(
                    ["playerctl", "metadata", "--format", "{{artist}} - {{title}}"],
                    check=True, text=True, capture_output=True
                )
                return result.stdout.strip()
            else:
                return f"Unknown command: {command}."

        return "Command executed successfully."
    except subprocess.CalledProcessError as e:
        return f"Error: {e}. No media player found or invalid command."
