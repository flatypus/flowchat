from flowchat import Chain, autodedent
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
            .link(autodedent(
                "Does the user want to exit the CLI? Respond with 'YES' or 'NO'.",
                user_input
            )).pull(max_tokens=2).unhook().last()
        )

        if should_exit.lower() in ("yes", "y"):
            print("Exiting the CLI.")
            break

        # Feed the input to flowchat
        command_suggestion = (
            Chain(model="gpt-4-1106-preview")
            .anchor(os_system_context)
            .link(autodedent(
                "The user wants to do this: ",
                user_input,
                "Suggest a command that can achieve this in one line without user input or interaction."
            )).pull().unhook()

            .anchor(os_system_context)
            .link(lambda suggestion: autodedent(
                "Extract ONLY the command from this command desciption:",
                suggestion
            ))
            # define a JSON schema to extract the command from the suggestion
            .pull(json_schema={"command": "echo 'Hello World!'"})
            .transform(lambda command_json: command_json["command"])
            .unhook().last()
        )

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
