def main():
    # Prompt the user for input
    user_input = input("Enter some text: ")

    # Open a file in write mode ('w')
    with open("output.txt", "w") as file:
        # Write the input to the file
        file.write(user_input)

    print("Input saved to output.txt")

if __name__ == "__main__":
    main()
