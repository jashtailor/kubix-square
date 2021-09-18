# kubix-square

- 10 Machine Learning models were trained on the .csv file uploaded on the repo.
- Out of those, the ones which had an accuracy greater than 60% were downloaded using the Pickle library. 
- In the final code for the frontend website, we have used the Coinbase API to get the list of the top 10 cryptocurrencies based on their market cap. 
- This list is then used to extract raw, live tweets from Twitter using the Twitter API.
- These tweets are preprocessed and fed into the selected 4 Machine Learning models. 
- Each model outputs a sentiment value (-1, 0, 1) for each tweet and the output from each model is compared and the majority is the final output. e.g if the output from all the 4 models is (0,1,1,1) then the final output is 1 i.e. Positive.
- The final output is displayed in a bar graph manner which the x-axis being the sentiments and y-axis being the number of tweets of each senitment.
 

Jash Tailor <br>
LinkedIn: https://www.linkedin.com/in/jashtailor/ <br>
Github: https://github.com/jashtailor

Beryl Coutinho <br>
LinkedIn: https://www.linkedin.com/in/beryl-coutinho/ <br>
Github: https://github.com/BerylCoutinho

https://share.streamlit.io/jashtailor/kubix-square/main/mk1.py
