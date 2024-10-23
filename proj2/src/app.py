from utils import *
nltk.download('stopwords')
nltk.download('punkt')
from flask import Flask, request, jsonify

app = Flask(__name__)


# input_corpus                        = Corpus()
# input_corpus.stemmed_data           = corpus_preprocessing()
# input_corpus.create_vocabulary()


@app.route('/execute_query', methods=['POST'])
def execute_query():
    global input_corpus
    try:
        data = request.get_json()
        # print(type(data))

        if 'queries' not in data:
            return jsonify({'error': "'queries' key is missing in the payload"}), 400

        result                          = {
                                            'postingsList'      : {},
                                            'postingsListSkip'  : {},
                                            'daatAnd'           : {},
                                            'daatAndSkip'       : {}    
                                        }
        
        queries                         = data['queries']
        # queries                         = {'queries': ["hello swimming", "hello world", "random swimming", "swimming going"]}
        # concatenated_queries            = " ".join(queries['queries'])
        concatenated_queries            = " ".join(queries)
        concatenated_queries            = query_preprocessing(concatenated_queries)
        concatenated_queries            = list(concatenated_queries)
        result['postingsList']          = get_postings_list_without_skip(concatenated_queries, input_corpus.vocabulary_mapping)
        result['postingsListSkip']      = get_postings_list_with_skip(concatenated_queries, input_corpus.vocabulary_mapping)

        for item in queries:
            queryset                        = query_preprocessing(item)
            queryset                        = list(set(queryset))

            result['daatAnd'][item]         = daat_without_skip(input_corpus.vocabulary_mapping, queryset)
            result['daatAndSkip'][item]     = daat_with_skip(input_corpus.vocabulary_mapping, queryset)


        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':

    input_corpus                        = Corpus()
    input_corpus.stemmed_data           = corpus_preprocessing()
    input_corpus.create_vocabulary()

    doc_token_count                     = {key: len(value) for item in input_corpus.stemmed_data for key, value in item.items()}
    set_tfidf(input_corpus.vocabulary_mapping, doc_token_count)

    for word, value in input_corpus.vocabulary_mapping.items():
        value.set_skip_pointers()

    app.run(host='0.0.0.0', port=9999)
