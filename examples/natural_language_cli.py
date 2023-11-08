from flowchat import Chain, autodedent
import json
import os
import subprocess


def execute_system_command(command):
    try:
        result = subprocess.run(
            command, shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr


def main():
    print("Welcome to the Natural Language Command Line Interface!")
    os_system_context = f"You are a shell interpreter assistant running on {os.name} operating system."

    while True:
        user_input = input("Please enter your command in natural language: ")

        should_exit = (
            Chain()
            .anchor("Does the user want to exit the CLI? Respond with 'YES' or 'NO'.")
            .link(f"User's response:\n{user_input}")
            .pull().unhook().last()
        )

        if should_exit.lower() in ("yes", "y"):
            print("Exiting the CLI.")
            break

        # Feed the input to flowchat
        command_suggestion_json = (
            Chain(model="gpt-4-1106-preview")
            .anchor(os_system_context)
            .link(autodedent(
                f"""
                The user wants to do this: 
                {user_input}
                Suggest a command that can achieve this in one line without user input or interaction.""")
            ).pull().unhook()

            .anchor(os_system_context)
            .link(lambda suggestion: autodedent(
                f"""
                Extract ONLY the command from this command desciption:
                {suggestion}

                Respond in the following example JSON format:
                {{
                    "command": "echo 'Hello World!'" 
                }}
                """
            ))
            .pull(
                response_format={"type": "json_object"},
            ).unhook().last()
        )

        command_suggestion = json.loads(command_suggestion_json)["command"]
        print(f"Suggested command: {command_suggestion}")

        # Execute the suggested command and get the result
        command_output = execute_system_command(command_suggestion)
        print(f"Command executed. Output:\n{command_output}")

        if command_output != "":
            description = (
                Chain().anchor(os_system_context)
                .link(f"Describe this output:\n{command_output}")
                .pull().unhook().last()
            )
            # Logging the description
            print(f"Explanation:\n{description}")

        print("=" * 60)


if __name__ == "__main__":
    main()
