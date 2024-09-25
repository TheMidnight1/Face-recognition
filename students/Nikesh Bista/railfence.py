def rail_fence_cipher(text, key, mode=' encrypt'):
    rail = [''] * key
    direction_down = False
    row = 0 
    if mode == 'encrypt':
        for char in text:
            rail[row] += char
            if row == 0 or row == key - 1:
                direction_down = not direction_down
            row += 1 if direction_down else -1
        return ''.join(rail)
    else:
        index = 0 
        result = [''] * len(text)
        for r in range(key):
            row = r
            direction_down = True if r == 0 else False
            while row < len(text):
                if row >= 0:
                    result[row] = text[index]
                    index += 1
                if r == 0 or r == key - 1:
                    row += 2 * (key - 1)
                else:
                    row += 2 * (key - 1 - r) if direction_down else 2 * r
                    direction_down = not direction_down
        return ''.join(result)

plaintext = input("Enter a plain text: ")
key = int(input("Enter number of rails: "))

encrypted_text = rail_fence_cipher(plaintext, key, mode = 'encrypt')
print("Encrypted:", encrypted_text)

decrypted_text = rail_fence_cipher(encrypted_text, key, mode = 'decrypt')
print("Decrypted:", decrypted_text)
