def ceaser_cipher(text, shift):
    result = ""
    for char in text:
        if char.isalpha():
            if char.isupper():
                shifted_char = chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
            else:
                shifted_char = chr((ord(char)- ord('a') + shift) % 26 + ord('a'))
            result += shifted_char
        else:
            result += char
    return result

plaintext = input("Enter a plain text: ")
shift = int(input("Enter the number of shifts: "))

encrypted_text = ceaser_cipher(plaintext, shift)
print("Encrypted:", encrypted_text)

decrypted_text = ceaser_cipher(encrypted_text, -shift)
print("Decrypted:", decrypted_text)