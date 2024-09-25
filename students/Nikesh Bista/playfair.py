def playfair_cipher(text, key, mode = "encrypt"):
    def prepare_key_table(key):
        key = ''.join(sorted(set(key), key = lambda x: key.index(x)))
        key_table = [char for char in key if char != 'J']
        alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'
        for char in alphabet:
            if char not in key_table:
                key_table.append(char)
        return [key_table[i * 5:(i + 1) * 5] for i in range(5)]
    
    def find_position(char, table):
        for i, row in enumerate(table):
            for j, letter in enumerate(row):
                if char == letter:
                    return i, j
        return None
    
    def process_digraph(digraph, table, mode):
        a, b = digraph
        row1, col1 = find_position(a, table)
        row2, col2 = find_position(b, table)
        if row1 == row2:
            if mode == 'encrypt':
                return table[row1][(col1 + 1) % 5] + table[row2][(col2 + 1) % 5]
            else:
                return table[row1][(col1 - 1) % 5] + table[row2][(col2 - 1) % 5]
        elif col1 == col2:
            if mode == "encrypt":
                return table[(row1 + 1) % 5][col1] + table[(row2 + 1) % 5][col2]
            else:
                return table[(row1 - 1) % 5][col1] + table[(row2 - 1) % 5][col2]
        else:
            return table[row1][col2] + table[row2][col1]
        
    def prepare_text(text):
        text = text.upper().replace('J', 'I')
        prepared_text = ""
        i = 0 
        while i < len(text):
            a = text[i]
            if i + 1 < len(text):
                b = text[i + 1]
            else:
                b = 'X'
            if a == b:
                prepared_text += a + 'X'
                i += 1
            else:
                prepared_text += a + b
                i += 2
        if len(prepared_text) % 2 != 0:
            prepared_text += 'X'
        return prepared_text
    
    key_table = prepare_key_table(key.upper())
    text = prepare_text(text)
    result = ''
    for i in range(0, len(text), 2):
        result += process_digraph(text[i: i + 2], key_table, mode)
    return result
plaintext = input("Enter a plain text: ")
key = input("Enter the key: ")
encrypted_text = playfair_cipher(plaintext, key, mode = 'encrypt')
print("Encrypted:", encrypted_text)
decrypted_text = playfair_cipher(encrypted_text, key, mode = 'decrypt')
print("Decrypted:", decrypted_text)




