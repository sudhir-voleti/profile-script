data = data.frame(id = 1:length(temp.text), text = temp.text, stringsAsFactors = F)
# dim(data)

# Read Stopwords list
stpw1 = readLines('https://raw.githubusercontent.com/sudhir-voleti/basic-text-analysis-shinyapp/master/data/stopwords.txt')      # stopwords list from git
stpw2 = tm::stopwords('english')                   # tm package stop word list; tokenizer package has the same name function
comn  = unique(c(stpw1, stpw2))                 # Union of two list
stopwords = unique(gsub("'"," ",comn))  # final stop word lsit after removing punctuation

x  = text.clean(data$text)             # pre-process text corpus
x  =  removeWords(x,stopwords)            # removing stopwords created above
x  =  stripWhitespace(x)                  # removing white space
# x  =  stemDocument(x)

#--------------------------------------------------------#
######  Create DTM using text2vec package                #
#--------------------------------------------------------#

t1 = Sys.time()

tok_fun = word_tokenizer

it_m = itoken(x,
              # preprocessor = text.clean,
              tokenizer = tok_fun,
              ids = data$id,
              progressbar = T)

vocab = create_vocabulary(it_m
                          # ngram = c(2L, 2L),
                          #stopwords = stopwords
                          )

pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 1)
# doc_proportion_max = 0.5,
# doc_proportion_min = 0.001)

vectorizer = vocab_vectorizer(pruned_vocab)

dtm_m  = create_dtm(it_m, vectorizer)
dim(dtm_m)

dtm = as.DocumentTermMatrix(dtm_m, weighting = weightTf)

print(difftime(Sys.time(), t1, units = 'sec'))


#------------------------------------------------------#
# Term Co-occurance Matrix                             #
#------------------------------------------------------#

vectorizer = vocab_vectorizer(pruned_vocab, grow_dtm = FALSE, skip_grams_window = 3L)
tcm = create_tcm(it_m, vectorizer)

tcm.mat = as.matrix(tcm)
adj.mat = tcm.mat + t(tcm.mat)
