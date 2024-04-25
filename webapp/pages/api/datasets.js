// Next.js API route support: https://nextjs.org/docs/api-routes/introduction

export default function handler(req, res) {
  const datasets = [
    {code: 'd1_fodors_zagats', name: 'Fodors-Zagats (Restaurants)'},
    {code: 'd2_abt_buy', name: 'Abt-Buy (Products)'},
    {code: 'd3_amazon_google', name: 'Amazon-Google (Products)'},
    {code: 'd4_dblp_acm', name: 'DBLP-ACM (Citations)'},
    {code: 'd5_imdb_tmdb', name: 'IMDB-TMDB (Movies)'},
    {code: 'd6_imdb_tvdb', name: 'IMDB-TVDB (Movies)'},
    {code: 'd7_tmdb_tvdb', name: 'TMDB-TVDB (Movies)'},
    {code: 'd8_amazon_walmart', name: 'Amazon-Walmart (Electronics)'},
    {code: 'd9_dblp_scholar', name: 'DBLP-Scholar (Citations)'},
    {code: 'd10_imdb_dbpedia', name: 'IMDB-DBPedia (Movies)'},
    {code: 'd11_itunes_amazon', name: 'iTunes-Amazon (Music)'},
    {code: 'd12_beeradvo_ratebeer', name: 'BeerAdvo-RateBeer (Beer)'},
  ];

  res.status(200).json(datasets);
}
