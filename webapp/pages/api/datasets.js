// Next.js API route support: https://nextjs.org/docs/api-routes/introduction

export default function handler(req, res) {
  const datasets = [
    {code: 'd1_fodors_zagats', name: 'Fodors-Zagats (Restaurants)'},
    {code: 'd2_abt_buy', name: 'Abt-Buy'},
    {code: 'd3_amazon_google', name: 'Amazon-Google (Products)'},
    {code: 'd4_dblp_acm', name: 'DBLP-ACM'},
    {code: 'd5_imdb_tmdb', name: 'IMDB-TMDB'},
    {code: 'd6_imdb_tvdb', name: 'IMDB-TVDB'},
    {code: 'd7_tmdb_tvdb', name: 'TMDB-TVDB'},
    {code: 'd8_amazon_walmart', name: 'Amazon-Walmart'},
    {code: 'd9_dblp_scholar', name: 'DBLP-Scholar'},
    {code: 'd10_imdb_dbpedia', name: 'IMDB-DBPedia (Movies)'},
  ];

  res.status(200).json(datasets);
}
