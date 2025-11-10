---
title: "Brewtiful: Embedding-Powered Beer Discovery Platform"
excerpt: "A modern beer discovery platform using vector embeddings"
header:
  teaser: /assets/images/brewtiful-teaser.png
toc: true
toc_label: "Contents"
toc_icon: "list"
toc_sticky: true
---

## What is Brewtiful?

**[Brewtiful]((https://www.brewtifulapp.com/))** is a beer discovery platform that helps users find new beers they'll love. With a catalog of over 60000+ beers from 6000+ breweries, the platform uses vector embeddings and machine learning to power recommendations help users find relevant items.

## The Tech Stack

Brewtiful is a modern full-stack application:

- **Frontend**: Next.js 15 with App Router, React, and TypeScript
- **Database**: PostgreSQL via Supabase with pgvector extension
- **Recommendations**: LightFM-learned embeddings served with k-means algorithm
- **Authentication**: Supabase Auth (Google OAuth)
- **Styling**: Tailwind CSS 3.4 with shadcn/ui components
- **Data Pipeline**: Scrapy + Playwright (python) for web scraping

## How Recommendations Work

### Using LightFM to Learn Embeddings

The recommendations are "powered" by embeddings learned by a [LightFM](https://making.lyst.com/lightfm/docs/home.html). LightFM is itself a derivative of the classic [Matrix Factorization](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)) (MF) technique. This post won't go into deep technical details about LightFM, but here is a brief crash course (skip this part if you are not interested in math). For a deep dive into the model training process, see this [Jupyter notebook](https://www.google.com) where I train a LightFM model on a public dataset. 

Classic MF models are often highly related to the concept of [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) from linear algebra. The SVD of any $n\times m$ real matrix $M$ factors it into the form $M=UDV^T$, where $U,V$ are orthogonal and of dim. $n \times n$ and $m \times m$ respectively, and $D$ is of rectangular diagonal of dim. $n \times m$. Let $k<m,n$ and consider $M_k = U_k D_k V_k^T$, where $D_k$ is a $k\times k$ matrix consisting of the $k$ largest values from $D$, and $U_k, V_k$ are $n \times k$, $m \times k$ and consist of the matrices "truncated" by removing the columns from $U,V$ corresponding to the values removed from $D$. It turns out that $M_k$ is the best $k$-ranked approximation of $M$ as measured by the Frobenius norm.

Now, consider the scenario of recommending items to users. Our data set takes the form of a $\vert P\vert\times \vert I\vert$ matrix $R$, where $P,I$ are our sets of users and items respectively (this matrix is called the **interaction matrix**). 

**The critical intuition**:\
*We can approximately represent both users and items in the same $k$-dimensional space!*

That is to say, we can "perform SVD" on the interaction matrix to obtain $k$-dimensional vectors for each item and user in our dataset, called the **item factors** and **user factors**. These embeddings compress the data contained in the interaction matrix and can be used in a variety of ways. In the following sections, I will explain how they are used to generate recommendations on Brewtiful. Note that we can't actually perform SVD on the interaction matrix, as most entries are empty (in practice, with large item catalogs, interaction matrices tend to be incredibly sparse). SVD models often use stochastic gradient descent or other numeric methods to compute the user and item factors.

The LightFM model seeks to expand on the simple SVD model by allowing for metadata features to learned as well. In the LightFM model, each user and item can be assigned features. Common user features may include age, location, gender, etc. For beer recommendation, item features may be brewery, style, IBU, etc. Each item and user is then represented in $k$ dimensions by the sum of their features, usually including an identity feature analogous to the embedding learned in the SVD model:

$$p_j = \sum_{i=0}^l f_{j_i}$$

where $f_j$ are the features attributed to $p_j$. There are multiple benefits to this approach. First, new items and users can still be given embeddings based off of their features, whereas traditional SVD models can only handle items they have "seen before". Second, including metadata features can sometimes improve the quality of recommendations (although this is not always the case; features must be chosen carefully).

Both LightFM and standard SVD models also learn *bias terms* to give a boost to high quality items. But how are the embeddings and bias terms used to serve recommendations? Before answering that question, we'll briefly digress the subject of the training data for the LightFM model.

#### Data Pipeline: From Web Scraping to Embeddings

Brewtiful's catalog was scraped (~2.5m data points) from publicly available sources using **Scrapy** with **Playwright**:

**Spider Configuration** (`webscraping/drunkenspiders/spiders/`):
- `BeerSpider`: Extracts name, style, ABV, description, ratings, `BrewerySpider`: Gathers location and active status, `ReviewSpider`: Scrapes user-item interactions with rating value, etc...
- Rate-limited and obeying `robots.txt` for respectful scraping

**Data Pipeline Flow**:
1. **Scrape** → Obtain raw JSON items from spiders
2. **Clean** → Jupyter notebooks normalize data (handle nulls, deduplicate, validate)
3. **Train** → LightFM model generates embeddings from rating matrix
4. **Upload** → Insert vectors into `beer_embeddings` table on Supabase

### Item-Item Similarity with pgvector

The simplest way to discover new beers is through **item-item similarity**. Each beer in the catalog has a 103-dimensional vector embedding stored in the `beer_embeddings` table (the item factors as learned by LightFM) in the Supabase PostgreSQL database. These embeddings capture characteristics like aggregated user preferences, style, and brewery.

When you view a beer's detail page, Brewtiful finds similar beers using PostgreSQL's **cosine similarity** via the pgvector extension:

```sql
CREATE OR REPLACE FUNCTION "public"."get_similar_beers"("beer_id_input" bigint, "match_count" integer DEFAULT 10, "show_inactive" boolean DEFAULT false) RETURNS TABLE("beer_id" bigint, "similarity" double precision)
    LANGUAGE "plpgsql"
    AS $$
BEGIN
    RETURN QUERY
    WITH ref_embedding AS (
        SELECT embedding
        FROM public.beer_embeddings
        WHERE id = beer_id_input
    )
    SELECT be.id, (ref_embedding.embedding <=> be.embedding) AS similarity
    FROM public.beer_embeddings AS be
    CROSS JOIN ref_embedding
    WHERE be.id != beer_id_input AND (show_inactive OR be.id IN (
      SELECT b.beer_id
      FROM public.beers as b
      WHERE active != 'Inactive'
      )
    )
    ORDER BY similarity, be.id
    LIMIT match_count;
END;
$$;
```
This function has a Typescript wrapper which is called by the client, but the results are fetched from the back-end via pgvector.

The database function uses an [**HNSW (Hierarchical Navigable Small World)**](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world) index on the embeddings for approximate nearest neighbor search, enabling sub-100ms queries across tens of thousands beers.

In short, the beers with embeddings pointing in the most similar direction (as measured by cosine similarity) are recommended as "similar items".

### Personalized Recommendations with K-Means Clustering

For users with at least 5 ratings, Brewtiful offers **personalized recommendations** balancing three competing objectives:

1. **Quality**: Recommend highly-rated beers
2. **Relevance**: Match the user's taste profile
3. **Diversity**: Avoid repetitive suggestions

The primary drivers of this recommendation model are the embeddings learned from LightFM. However, these recommendations are served using a weighted K-means clustering approach, not cosine similarity like item-item recommendations.

Here's how it works:

**Step 1: Gather User Data**

The user's rating history is collected from the database.

```typescript
// Fetch user's rated beers with their embeddings
const { data: ratings } = await supabase
  .from('user_ratings')
  .select(`
    rating,
    beer:beers (
      beer_id,
      name,
      bias_term,
      embedding:beer_embeddings (embedding)
    )
  `)
  .eq('user_id', userId);
```

**Step 2: Cluster with Weighted K-Means**

The algorithm learns clusters of beer embeddings based on the user's rating history. Higher-rated beers have more influence on cluster centroid positions:

```typescript
// Simplified weighted K-means iteration
centroids[clusterIdx] = beers
  .filter(b => b.cluster === clusterIdx)
  .reduce((sum, beer) => {
    const weight = beer.rating / totalWeight;
    return sum.map((v, i) => v + beer.embedding[i] * weight);
  }, new Array(103).fill(0));
```

This is deterministic—the same ratings produce identical clusters using a seeded random number generator.

**Step 3: Fetch Candidates per Cluster**

For each learned cluster centroid, call the database to find similar beers:

```typescript
const { data } = await supabase.rpc('get_beers_similar_to_centroids', {
  centroids: clusterCentroids,
  match_count: 100,
  show_inactive: false
});
```

**Step 4: Rank with Diversity**

We collect the beers most similar to the style centroids calculated via k-means, but re-rank them for a high-quality user experience. The final ranking balances quality, similarity, and diversity using three hyperparameters:

```typescript
// Tunable via environment variables
const config = {
  alpha: 0.1,      // Quality influence (bias_term weight)
  lambda: 0.1,     // Inter-cluster diversity decay
  topK: 5          // Items for cluster quality averaging
};
```

The ranking algorithm:
- Prioritizes beers from high-quality clusters (average `bias_term` of top K items)
- Decays diversity penalty across different clusters to encourage variety

This produces a final recommendation list that's both relevant and reflects the spectrum of a user's taste.

### Lessons Learned
#### Serving Methods
It took me a while to arrive at the k-means implementation. Since Brewtiful is starting out as a free project, I have limited access to server-side compute. Therefore, to serve recommendations, I needed simple methods which relied mostly on javascript and pgvector similarity calculations. Retraining the LightFM model constantly was out of the question. Initially, an attractively simple method - users would be represented by a weighted sum of the item embeddings they had rated:

$$ u = \sum_{i=0}^k w_i q_i $$

where $q_i$ are the items rated by user $u$ and $w_i$ is the weight assigned based on their rating. Recommendations would then be served via cosine similarity over the item catalog. In practice, I found (heuristically) that after several ratings, the user vector would tend to get "bloated" and all user preference signal would be lost - rating additional items would not change the beers recommended to the user. The k-means algorithm offers a huge improvement by allowing users' numerous taste profiles to be reflected in their recommendations instead of trying to fit all user preference signal into one vector. Crucially, this approach is not very computationally intensive.

#### Bias Terms
It's also worth noting that bias terms need to be monitored carefully. When bias terms are too large, the recommender tends to favour certain highly popular beers all the time instead of adapting to the user's tastes. While bias terms can be important for recommending quality items, it's important to not let niche items get swamped.

## Web Architecture
### Server Components and Performance

Next.js 15's App Router allows Brewtiful to colocate data fetching with UI rendering. Here's a typical beer detail page:

```typescript
// app/beers/[id]/page.tsx (Server Component)
export default async function BeerPage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ showInactive?: string }>;
}) {
  const { id } = await params;
  const { showInactive } = await searchParams;
  const beerId = parseInt(id);

  const supabase = await createClient();

  // Fetch beer with brewery details (single query with join)
  const { data: beer } = await supabase
    .from('beers')
    .select(`
      *,
      brewery:breweries (brewery_id, name, city, country)
    `)
    .eq('beer_id', beerId)
    .single();

  // Get similar beers via RPC
  const similarBeers = await getSimilarBeers(
    beerId,
    10,
    showInactive === 'true'
  );

  return (
    <>
      <BeerInfoCard beer={beer} />
      <SimilarBeers beerId={beerId} initialBeers={similarBeers} />
    </>
  );
}
```

**Performance optimizations**:
- **HNSW indexes** for <100ms vector queries
- **Composite indexes** on frequently filtered columns (style, brewery_id, active status) 
- Bulk data fetching and caching when possible

### Authentication Flow

Brewtiful uses **Supabase Auth** with a cookie-based session model:

**Google OAuth** (primary method):
1. User clicks "Sign in with Google"
2. Redirect to Google OAuth consent screen
3. Callback handler (`/app/auth/callback/route.ts`) exchanges code for session
4. Session stored in httpOnly cookies
5. Middleware refreshes session on every request

## Future Improvements?

Brewtiful is a hobby project and it's unlikely I will continue to improve it much further.

That being said, I learned many things working on it, including the steps I would take to scale it:

**Cold Start Items**:
- One of the biggest advantages of LightFM models is that you can use feature embeddings to give new items an embedding representation without interaction data. I have not exploited that fact for Brewtiful - all the items on brewtiful were items from the filtered dataset. There were many more items I could have generated embeddings for. I did not due so due to time constraints, but it would be easy to implement.

**Proper Evaluation for K-means**:
- The current model training notebooks (see link at bottom of page) evaluate the LightFM models on offline metrics, but the production model serves recommendations with a hybrid k-means system that is not evaluated. It would be interesting and useful to evaluate the actual production model (it would be even more interesting to evaluate its online performance).

**User Embeddings**:
- Generate persistent user vectors from rating history by training LightFM model on new interaction data. These user vectors could be used to improve the recommendation model (for example, reranking k-means results via user factor cosine similarity score).
- Support cold-start recommendations for new users (for example, by learning features for user location).

**Deepen Recommendation/Reranking Model**:
- Incorporate user event tracking data to model (page clicks, searches, etc.)
- Incorporate temporal signals (seasonal beers, trending styles)

The `events` table already captures user interactions (`rate`, `view`, `search`, `get_recs`, `save`, `unsave`) with a flexible `jsonb` metadata field. This logging infrastructure could power future model iterations and product analytics (implicit interaction data).

**MLOps**:
With cloud computing, it would be possible to create a data pipeline to load user interaction/rating data from supabase, automatically retrain user and item embeddings, and use them to serve recommendations (while also implementing models to dynamically update recommendations as they are now).

**User Experience Testing**:
- A/B test for tuning various recommendation hyperparameters with metrics like CTR, precision, etc.
- Allow users to try "Discovery Mode", with increased focus on item diversity (less weight on bias term, penalize intra-cluster similarity, etc.)

## Try It Yourself

Everything is open source. Feel free to check out the codebase [here](https://github.com/lucaslyons/brewtiful).

---

*Want to dive deeper? Check out the [Jupyter notebook on LightFM model training](https://github.com/LucasLyons/Brewtiful/tree/main/brewtiful-fm-demo).*
