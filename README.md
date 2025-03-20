# Company Classifier

In this document, I will present my thought process. If you want to see the solution for the challenge, you can find it in the Jupyter notebook `Company_Classifier`.

In the CSV folder, you will find two directories: one called `Data`, which contains the untouched data and the cleaned data that I used, and one called `Tries`, where I stored all the predictions that I made.

For the coding part, you'll find everything in the `Code` folder, where you'll be able to see all of my attempts and some visualizations of the solutions I thought were the best.

## My thought process

Firstly, after I read the requirements, I started looking into the `company_list` file to better understand the data. From the first look, I decided that the information in the `sector` column would not really help me, so I did not took it into account.

Next, I realized that I didn’t know much about solving NLP problems, so I started looking online for general steps that I may use(https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/)

Then I started: removing punctuation, converting text to lowercase, tokenizing, and removing stop word. At that point, I noticed that the data still had a lot of information that wouldn’t help me. So, I decided to remove things like common words, cities, countries, or even company names from the descriptions (including non-English words). After doing this, I lemmatized everything.

At this point, I was looking for a way to analyze and predict the labels for each company. My initial idea was to find synonyms for the insurance taxonomy and then search for those synonyms in every column (except `sector`).where i would find the synonimes. However, my biggest problem was that I couldn’t find an efficient way to do this. Whether I scraped synonyms from the Cambridge Dictionary or used an existing database, neither seemed like a good choice.

So, I asked a friend what kind of solution they would approach. That’s how I found out about `symmetric similarity`. The next step was to research this concept to make the best use of it (https://www.newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python)

Now I started trying different models, deciding to use in this case cosine similarity. The first one I tried was Word2Vec, which didn’t seem efficient (it took nearly 80 minutes to calculate the similarities), so I did not continue with the idea due to the time inefficiency.

Next was Doc2vec(Dbow), which was more promising. At this point, I read about how important context is in the data and how you can train the model to treat the data as sentences, not just words. Knowing that, I redid the data preparation without removing the most common words, without lemmatization and withput tokenizing.

Initially, since I was categorizing each company with labels and using those labels for similarity, DBOW seemed promising. However, when I looked at the predictions, the assigned labels were far from correct. 

![Output DBOW](Images/outputDBOW.png.png "Output")  

Looking at this, we can see that for the first 50 companies, almost all of them had high similarity scores for the first 50 labels, which shouldn’t have happened. The issue might have been caused by how the model works, where for each corpus, a random sentence is selected with a random number of words, and those words are not always the best.

The next approach was sentence transformers, like BERT and SBERT:  
https://www.sbert.net/docs/sentence_transformer/pretrained_models.html  
https://huggingface.co/sentence-transformers  

I found some interesting models to try, like `all-mpnet-base-v2` (`categorizing_sbert1`) and `all-MiniLM-L6-v2` (`categorizing_sbert2`). Latter, I decided to go with the second one as the final result.

The first model didn’t produce a lot of different results compared to the second one(from what i saw looking manually inside), but the worst part was the time it took(approximately 15 minutes,at least). So, I chose the second one, even though its similarities scores were not the highest.

Since I wasn’t completely happy with the results, I looked for more information about what I needed to do.

That’s how I found information about training a model (https://sbert.net/docs/sentence_transformer/training_overview.html#dataset). I also got curious about the loss function, specifically the Loss Overview (https://sbert.net/docs/sentence_transformer/loss_overview.html). Knowing that I needed to find the pair (company description or business tags and label) with the highest similarity score, I went back to cosine similarity loss. While reading about it, I found information about semantic search (https://www.sbert.net/examples/applications/semantic-search/README.html).


Next, I decided to look on Hugging Face for pre-trained models suitable for categorizing. I found two more models:  
- `bert-base-nli-mean-tokens` (`categorizing_bert`)  
- `distilbert-base-uncased` (`categorizing_distilbert`)  

I chose to try the first one because it seemed slightly different (though not the best choice) and the second one because it seemed promising. In these cases, the first model had low similarity overall and assigned many labels to the same company. The second model had a good similarity score but assigned even more irrelevant labels.

After trying symmetric search methods, I thought it might not break anything if to treat the company descriptions as questions and the labels as answers. So, I tried asymmetric semantic search using the model `msmarco-MiniLM-L6-cos-v5` (`categorizing_asymmetric_search`). Initially, the results seemed good, but after reviewing the assigned labels, I realized it was the worst approach.

The last attempt was using the Universal Sentence Encoder (https://tfhub.dev/google/universal-sentence-encoder/4) (`categorizing_USE`). The similarity score was low, and while some data seemed appropriate, most of it didn’t assign even one good label for categorization.

## Choosing the best model

After trying all the models above, I looked for a way to verify the results. I started by searching the company names in the descriptions. It didn’t take long to realize that the company descriptions were scraped from their own websites. I wanted to do the same, scrape the internet for sites with insurance taxonomies for each company. Unfortunately, I couldn’t find a suitable website to start with.

Since I couldn’t find a practical way to do this, I decided to verify the results manually. Of course, I didn’t do this for the entire dataset but for a sample of it. After analyzing the results of the seven attempts that produced something, the best models were the sentence-BERT ones: `all-mpnet-base-v2` and `all-MiniLM-L6-v2`. Between these two, the second one was the winner because it required less execution time.

## Biggest problems of the solution

To make sure the model produced good results, I wanted it to classify each company into the top 3 insurance labels. However, the problem here is that some companies might belong to fewer than three categories, while others might belong to more.

Another major issue is that I don’t have an automatic way to check the results, so I need human input to verify over 9000 types of data, which is not ideal.

The longer the dataset, the more time the model takes to process everything.

Lastly, the inconsistent quality of the data provided for all companies can cause problems. If we clean a description too much, we might end up with only one word, which may not be relevant. The same applies to other types of information provided, like niche or category.

