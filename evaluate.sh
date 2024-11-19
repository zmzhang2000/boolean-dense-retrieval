set -e

for DATA_NAME in "MSMARCO" "NaturalQuestions"; do
    if [ $DATA_NAME == "MSMARCO" ]; then
        QUERY_ENCODERS=("sentence-transformers/msmarco-distilbert-cos-v5"
                        "sentence-transformers/msmarco-MiniLM-L12-cos-v5"
                        "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
                        "sentence-transformers/msmarco-distilbert-dot-v5"
                        "sentence-transformers/msmarco-bert-base-dot-v5"
                        "sentence-transformers/msmarco-distilbert-base-tas-b"
                        "intfloat/e5-base-v2"
                        "intfloat/e5-large-v2"
                        "BAAI/bge-large-en-v1.5"
                        "Alibaba-NLP/gte-Qwen2-7B-instruct")
    elif [ $DATA_NAME == "NaturalQuestions" ]; then
        QUERY_ENCODERS=("sentence-transformers/nq-distilbert-base-v1"
                        "sentence-transformers/facebook-dpr-question_encoder-single-nq-base"
                        "dunzhang/stella_en_1.5B_v5"
                        "BAAI/bge-large-en-v1.5"
                        "intfloat/e5-base-v2"
                        "intfloat/e5-large-v2"
                        "Alibaba-NLP/gte-Qwen2-7B-instruct")
    else
        echo "Invalid data name"
        continue
    fi

    for QUERY_ENCODER in "${QUERY_ENCODERS[@]}"; do
        if [ $QUERY_ENCODER == "sentence-transformers/facebook-dpr-question_encoder-single-nq-base" ]; then
            DOCUMENT_ENCODER="sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base"
        else
            DOCUMENT_ENCODER=$QUERY_ENCODER
        fi

        echo "Evaluating $DATA_NAME with encoders: [QUERY] $QUERY_ENCODER and [DOCUMENT] $DOCUMENT_ENCODER"

        huggingface-cli download $QUERY_ENCODER
        huggingface-cli download $DOCUMENT_ENCODER

        EXP_NAME=BoolQuestions_${DATA_NAME}_${QUERY_ENCODER}
        EXP_NAME=${EXP_NAME//\//-}

        bash generate_embeddings.sh $QUERY_ENCODER $DOCUMENT_ENCODER $DATA_NAME output/$EXP_NAME
        python retrieve.py --input_path output/$EXP_NAME --output_path output/$EXP_NAME/retrieval --encode_queries_online --queries_path $DATA_NAME --query_encoder $QUERY_ENCODER
        python evaluate.py --results_file output/$EXP_NAME/retrieval/results.top100.txt --data $DATA_NAME
    done
done