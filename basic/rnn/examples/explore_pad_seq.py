from keras_preprocessing.sequence import pad_sequences

def main():
    data = [
        [1,2,3],
        [1,2,3,4],
        [1]
    ]
    # 아, padding = 'pre' 일때는 앞부분에 붙는구나.
    data_padding = pad_sequences(sequences=data, padding='pre', value=0)
    print(data_padding)
    data_padding = pad_sequences(sequences=data, padding='post', value=0)
    print(data_padding)
    data_padding = pad_sequences(sequences=data, padding='post', value=1231241252)
    print(data_padding)
    
if __name__ == '__main__':
    main()