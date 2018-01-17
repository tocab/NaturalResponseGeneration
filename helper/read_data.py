import html, re, json

# Regex for space between symbols
symbols_re = re.compile(r"([-!?.:'*´`,+/\\=(){}&%$§\"])")

# html regex
html_re = re.compile(r"<.*?>")

# Clean whitespace
whitespace_re = re.compile(r"\s+")

# Repeating char regex
rep_char_re = re.compile(r"((.)\2{2})\2+")


# Expand contractions
def load_contractions():
    """
    Loads the contractions lexicon from file
    source: http://devpost.com/software/contraction-expander
    """

    # Open contractions json file
    with open("res/contractions_expanded.json") as f:
        data = f.read()

    # Read data
    cList = json.loads(data)
    return cList


# Handle contractions lexicon
contractions = load_contractions()


def load_vocab_and_dict(data_folder):
    """
    Load vocab file and bpe file from preprocessed data
    :param data_folder: location of preprocessed data
    :return: bpe file and vocab list
    """
    # Load vocab and bpe dict
    vocab_list = []
    bpe_dict = {}
    with open(data_folder + "vocab.vocab", "r", encoding="utf-8") as f:
        for line in f:
            vocab_list.append(line.strip())
    with open(data_folder + "bpe_assign.vocab", "r", encoding="utf-8") as f:
        for line in f:
            splitted = line.split("\t")
            bpe_dict[splitted[0].strip()] = splitted[1].strip()

    return bpe_dict, vocab_list


def process_text(text):
    """
    Text cleaning step with regular expressions and contractions dict
    :param text: input text
    :return: cleaned text
    """
    # Unescape and remove html characters
    text = html.unescape(text)
    text = html_re.sub(r" ", text)

    # Lower all characters
    text = text.lower()

    # Expand contractions
    for key in contractions:
        text = text.replace(key, contractions[key])

    # Replace repeating characters
    text = rep_char_re.sub(r"\1", text)

    # space between terminal symbols and words
    text = symbols_re.sub(r" \1 ", text)

    # Clean whitespace
    text = whitespace_re.sub(r" ", text)

    return text


def process_questions(questions, bpe_dict, vocab_list):
    """
    Processes an input question
    :param questions: Input question as text
    :param bpe_dict: bpe dict out of preprocessed data
    :param vocab_list: vocab list out of preprocessed data
    :return: processed questions and corresponding length
    """

    # Initialize list for input length and processed questions
    input_length = []
    questions_processed = []

    # for every question in questions
    for q in questions:

        # Process question text
        line_words = process_text(q)

        # Split words on whitespace
        line_words = line_words.split()

        # look up words splits with bpe dict
        line_words = [bpe_dict[word].split() if word in bpe_dict else [word] for word in line_words]

        # Flatten list
        line_words = [item for sublist in line_words for item in sublist]

        # Determine length of sequence
        input_length.append(len(line_words))

        # look up word in vocab list. If word is not found, mark it with UNK token
        line_seq = [
            vocab_list.index(word) if word in vocab_list else vocab_list.index("<UNK>") for
            word in line_words]
        questions_processed.append(line_seq)
    return questions_processed, input_length
