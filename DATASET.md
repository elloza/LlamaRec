# ml_100k
wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip

Links.csv
```
movieId,imdbId,tmdbId
1,0114709,862
2,0113497,8844
```

movies.csv
```
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
```

ratings.csv
```
userId,movieId,rating,timestamp
1,1,4.0,964982703
1,3,4.0,964981247
```

tags.csv
```
userId,movieId,tag,timestamp
2,60756,funny,1445714994
2,60756,Highly quotable,1445714996
```

# Beauty
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv 
gunzip meta_Beauty.json.gz

Ratings.csv
```
A39HTATAQ9V7YF,0205616461,5.0,1369699200
A3JM6GV9MNOF9X,0558925278,3.0,1355443200
...
```
meta.json
```json
{
    'asin': '0558925278', 
    'description': 'Mineral Powder Brush--Apply powder or mineral foundation all over face in a circular, buffing motion and work inward towards nose.\n\nConcealer Brush--Use with liquid or mineral powder concealer for more coverage on blemishes and under eyes. \n\nEye Shading Brush-- Expertly cut to apply and blend powder eye shadows.\n\nBaby Kabuki-- Buff powder over areas that need more coverage. \n\nCosmetic Brush Bag-- 55% hemp linen, 45% cotton', 'title': 'Eco Friendly Ecotools Quality Natural Bamboo Cosmetic Mineral Brush Set Kit of 4 Soft Brushes and 1 Pouch Baby Kabuki Eye Shading Brush Mineral Powder Brush Concealer Brush(travle Size)', 
    'imUrl': 'http://ecx.images-amazon.com/images/I/51L%2BzYCQWSL._SX300_.jpg', 
    'salesRank': {'Beauty': 402875}, 
    'categories': [['Beauty', 'Tools & Accessories', 'Makeup Brushes & Tools', 'Brushes & Applicators']]
},
{
    'asin': '0737104473',
    'description': 'Limited edition Hello Kitty Lipstick featuring shiny black casing with Hello Kitty figure on a pop art pattern background. Cap features the logos of both MAC and Hello Kitty in this collection.',
    'title': 'Hello Kitty Lustre Lipstick (See sellers comments for colors)',
    'imUrl': 'http://ecx.images-amazon.com/images/I/31u6Hrzk3WL._SY300_.jpg',
    'salesRank': {'Beauty': 931125},
    'categories': [['Beauty', 'Makeup', 'Lips', 'Lipstick']]
}
...
```

# Video Games
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Video_Games.csv
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz
gunzip meta_Video_Games.json.gz

```
A24SSUT5CSW8BH,0078764343,5.0,1377302400
AK3V0HEBJMQ7J,0078764343,4.0,1372896000
...
```

```json
{
    "category": ["Video Games", "PC", "Games"], 
    "tech1": "",
     "description": [], 
     "fit": "", 
     "title": "Reversi Sensory Challenger", 
     "also_buy": [], 
     "tech2": "", 
     "brand": "Fidelity Electronics", 
     "feature": [], 
     "rank": [">#2,623,937 in Toys &amp; Games (See Top 100 in Toys &amp; Games)", ">#39,015 in Video Games &gt; PC Games"], 
     "also_view": [], 
     "main_cat": "Toys &amp; Games", 
     "similar_item": "", 
     "date": "", 
     "price": "", 
     "asin": "0042000742", 
     "imageURL": ["https://images-na.ssl-images-amazon.com/images/I/31nTxlNh1OL._SS40_.jpg"], 
     "imageURLHighRes": ["https://images-na.ssl-images-amazon.com/images/I/31nTxlNh1OL.jpg"]
}
{
    "category": ["Video Games", "PC", "Games"],
    "tech1": "",
    "description": [],
    "fit": "",
    "title": "Reversi Sensory Challenger",
    "also_buy": [],
    "tech2": "",
    "brand": "Fidelity Electronics",
    "feature": [],
    "rank": [">#2,623,937 in Toys &amp; Games (See Top 100 in Toys &amp; Games)", ">#39,015 in Video Games &gt; PC Games"], 
    "also_view": [],
    "main_cat": "Toys &amp; Games",
    "similar_item": "",
    "date": "",
    "price": "",
    "asin": "0042000742",
    "imageURL": ["https://images-na.ssl-images-amazon.com/images/I/31nTxlNh1OL._SS40_.jpg"],
    "imageURLHighRes": ["https://images-na.ssl-images-amazon.com/images/I/31nTxlNh1OL.jpg"]
}
```