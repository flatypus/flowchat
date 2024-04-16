from flowchat import Chain, autodedent
import os
import subprocess


def execute_system_command(command: str) -> str:
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

        # ========================================================================== #

        should_exit = (
            Chain(model="gpt-3.5-turbo")
            .link(autodedent(
                "Does the user want to exit the CLI? Respond with 'YES' or 'NO'.",
                user_input
            )).pull(max_tokens=2).unhook().last()
        )

        if should_exit.lower() in ("yes", "y"):
            print("Exiting the CLI.")
            break

        # ========================================================================== #

        print("Checking if the command is possible to execute...")

        # Check if the user's request is possible; example of nested chains!
        # In practice, you could just ignore all of this and just execute the command.

        possible = (
            Chain(model="gpt-4-turbo")
            .anchor(os_system_context)
            .link(autodedent(
                "The user would like to do this: ",
                user_input
            ))
            .link("Create a short list of the minimum requirements that need to be checked in order to determine if this action is possible on this device.")
            .pull(json_schema={"requirement_list": "List[str]"})
            .transform(lambda requirement_json: requirement_json["requirement_list"]).log()
            .transform(lambda requirement_list: [
                Chain("gpt-4-turbo")
                .anchor(os_system_context)
                .link(autodedent(
                    "Suggest a command that can check if this requirement is met. The command should be a one-liner without user input or interaction.",
                    requirement,
                    "If the command needs additional information, you can include it. If the command itself can be run alone, leave additional_info an empty list."
                ))
                .pull(json_schema={"command": "string", "additional_info": "List[str]"})
                .transform(lambda command_json: (command_json["command"], [
                    Chain("gpt-4-turbo")
                    .anchor(os_system_context)
                    .link(autodedent(
                        "The user would like to know this information: ",
                        info,
                        "Suggest a command that can check if this information is available."
                    ))
                    .pull(json_schema={"command": "string"})
                    .transform(lambda command_json: command_json["command"])
                    .transform(lambda command: f"{info} | Output:{execute_system_command(command)}")
                    .unhook().last()
                    for info in command_json.get("additional_info")]
                )).unhook()

                .anchor(os_system_context)
                .link(lambda command_info: autodedent(
                    "Include the additional information in the command:",
                    command_info[0],
                    *command_info[1],
                    "to create a final command that can check if this requirement is met:",
                    requirement
                ))
                .pull(json_schema={"command": "string"})
                .transform(lambda command_json: command_json["command"])
                .unhook()

                .anchor(os_system_context)
                .transform(
                    lambda command: Chain("gpt-4-turbo")
                    .anchor(os_system_context)
                    .link(autodedent(
                        f"The user would like to check if this requirement is met: {requirement}",
                        f"The user executes this command: {command}:",
                        "Output:",
                        (lambda a: a if a else "<empty_response>")(
                            execute_system_command(command)
                        ),
                        f"Does the output indicate that the requirement is met?",
                    ))
                    .pull(json_schema={"is_met": "bool"})
                    .transform(lambda is_met_json: is_met_json["is_met"])
                    .unhook().last()
                )
                .last()
                for requirement in requirement_list
            ])
            .last()
        )

        if all(possible):
            print("This command should be possible to execute!")
        elif sum(possible) / len(possible) > 0.5:
            print("This command might be possible to execute.")
        else:
            print("This command is not possible to execute.")
            continue

        # ========================================================================== #

        print("Suggesting a command...")

        # Feed the input to flowchat
        command_suggestion = (
            Chain(model="gpt-4-turbo")
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

        # ========================================================================== #

        # Execute the suggested command and get the result
        command_output = execute_system_command(command_suggestion)
        print(f"Command executed. Output:\n{command_output}")

        if command_output != "":
            description = (
                Chain(model="gpt-3.5-turbo").anchor(os_system_context)
                .link(f"Describe this output:\n{command_output}")
                .pull().unhook().last()
            )
            # Logging the description
            print(f"Explanation:\n{description}")

        print("=" * 60)


if __name__ == "__main__":
    main()
