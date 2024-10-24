from utils import *
nltk.download('stopwords')
nltk.download('punkt')
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/execute_query', methods=['POST'])
def execute_query():
    global input_corpus
    global doc_scores
    try:
        data = request.get_json()
        # print(type(data))

        if 'queries' not in data:
            return jsonify({'error': "'queries' key is missing in the payload"}), 400

        result                          = {
                                            'postingsList'      : {},
                                            'postingsListSkip'  : {},
                                            'daatAnd'           : {},
                                            'daatAndSkip'       : {}, 
                                            'daatAndTfIdf'      : {},
                                            'daatAndSkipTfIdf'  : {}, 
                                        }
        
        queries                         = data['queries']
        # queries                         = {'queries': ["hello swimming", "hello world", "random swimming", "swimming going"]}
        # concatenated_queries            = " ".join(queries['queries'])
        concatenated_queries            = " ".join(queries)
        concatenated_queries            = query_preprocessing(concatenated_queries)
        concatenated_queries            = list(concatenated_queries)

        result['postingsList']          = get_postings_list_without_skip(concatenated_queries, input_corpus.vocabulary_mapping)
        result['postingsList']          = { term: [ptr.data for ptr in posting_list] for term, posting_list in result['postingsList'].items()}

        result['postingsListSkip']      = get_postings_list_with_skip(concatenated_queries, input_corpus.vocabulary_mapping)
        result['postingsListSkip']      = { term: [ptr.data for ptr in posting_list] for term, posting_list in result['postingsListSkip'].items()}

        for item in queries:
            queryset                            = query_preprocessing(item)
            queryset                            = list(set(queryset))

            result['daatAnd'][item]                 = daat_without_skip_more_than_2_terms(input_corpus.vocabulary_mapping, queryset)
            result['daatAndTfIdf'][item]            = daat_without_skip_more_than_2_terms(input_corpus.vocabulary_mapping, queryset)

            result['daatAnd'][item]['results']      = [ptr.data for ptr in result['daatAnd'][item]['results']]        
            result['daatAndTfIdf'][item]['results'] = sort_on_TfIdf(result['daatAndTfIdf'][item]['results'], doc_scores)
            result['daatAndTfIdf'][item]['results'] = [ptr.data for ptr in result['daatAndTfIdf'][item]['results']]

            result['daatAndSkip'][item]             = daat_with_skip(input_corpus.vocabulary_mapping, queryset)
            result['daatAndSkipTfIdf'][item]        = daat_with_skip(input_corpus.vocabulary_mapping, queryset)

            result['daatAndSkip'][item]['results']      = [ptr.data for ptr in result['daatAndSkip'][item]['results']]   
            result['daatAndSkipTfIdf'][item]['results'] = sort_on_TfIdf(result['daatAndSkipTfIdf'][item]['results'], doc_scores)
            result['daatAndSkipTfIdf'][item]['results'] = [ptr.data for ptr in result['daatAndSkipTfIdf'][item]['results']]

        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':

    input_corpus                        = Corpus()
    input_corpus.stemmed_data           = corpus_preprocessing()
    input_corpus.create_vocabulary()

    for word, value in input_corpus.vocabulary_mapping.items():
        value.set_skip_pointers()

    doc_token_count                     = {key: len(value) for item in input_corpus.stemmed_data for key, value in item.items()}
    doc_scores                          = {key: Document(key) for item in input_corpus.stemmed_data for key, value in item.items()}

    set_tfidf(input_corpus.vocabulary_mapping, doc_token_count)
    doc_scores                          = set_document_scores(input_corpus.vocabulary_mapping, doc_scores)

    app.run(host='0.0.0.0', port=9999)
