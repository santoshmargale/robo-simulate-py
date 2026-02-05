l = 2; b = 6
print('Area of rectangle:', l * b)
import random

def generate_number():
    num = random.randint(1000, 9999)
    return [int(x) for x in str(num)]

def input_guess():
    guess = input("Enter your guess: ")
    return [int(i) for i in guess]

def start_game():
    tries = 0
    number = generate_number()
    while tries < 10:
        result = ""
        guess = input_guess()
        if len(guess) != 4:
            print("Enter only 4 digits.")
            continue
        if guess == number:
            print(f"You won with {tries} attempts.")
            break
        for index, element in enumerate(guess):
            if element == number[index]:
                result += "C"
            elif element in number:
                result += "B"
            else:
                result += "A"
        print(result)
        tries += 1
    else:
        print(f"You ran out of attempts. The answer was: {''.join(map(str, number))}")

start_game()   



# A simple number guessing game
import random

def guess_game():
    print("Welcome to the Number Guessing Game!")
    print("I'm thinking of a number between 1 and 100.")
    
    secret_number = random.randint(1, 100)
    attempts = 0
    max_attempts = 7

    while attempts < max_attempts:
        try:
            guess = int(input(f"Enter your guess (attempt {attempts + 1}/{max_attempts}): "))
            attempts += 1

            if guess < secret_number:
                print("Too low! Try a higher number.")
            elif guess > secret_number:
                print("Too high! Try a lower number.")
            else:
                print(f"🎉 Congratulations! You guessed the number in {attempts} attempts!")
                return
        except ValueError:
            print("Please enter a valid integer.")
            attempts -= 1  # Don't count invalid input as an attempt

    print(f"❌ Sorry, you've used all {max_attempts} attempts. The number was {secret_number}.")

# Start the game
if __name__ == "__main__":
    guess_game()   




