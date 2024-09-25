def vignere_cipher(text, key, mode="encrypt"):
    key = key.upper()
    result = []
    key_index = 0
    for char in text:
        if char.isalpha():
            shift = ord(key[key_index % len(key)]) - ord('A')
            if mode == "decrypt":
                shift = -shift
            if char.isupper():
                shifted_char = chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
            else:
                shifted_char = chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
            key_index += 1
        else:
            shifted_char = char
        result.append(shifted_char)
    return ''.join(result)

plaintext = input("Enter a plain text: ")
key = input("Enter the key: ")

encrypted_text = vignere_cipher(plaintext, key, mode = 'encrypt')
print("Encrypted:", encrypted_text)

decrypted_text = vignere_cipher(encrypted_text, key, mode = 'decrypt')
print("Decrypted:", decrypted_text)
