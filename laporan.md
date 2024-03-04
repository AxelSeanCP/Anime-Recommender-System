# Laporan Proyek Machine Learning - Axel Sean Cahyono Putra

# Anime Recommender System

## Project Overview
Di zaman digital seperti ini tentu banyak informasi dan hiburan yang beredar, contohnya adalah video game, film hollywood, e-book novel atau komik, dan tentu saja film animasi atau bisa disebut **anime**. Karena banyaknya jenis hiburan tersebut pasti tidak sedikit orang yang suka menonton anime, diantara orang orang itu pun pasti juga memiliki selera yang berbeda dalam menonton anime. 

Karena selera yang beragam tentu orang-orang suka membentuk suatu komunitas anime untuk membahas tentang anime yang mereka tonton atau untuk meminta rekomendasi anime untuk ditonton selanjutnya. Namun terkadang rekomendasi yang diberikan tidak selalu cocok karena tiap orang pasti punya selera yang berbeda, contoh orang "A" suka anime dengan tema **fantasy** dan **adventure** lalu meminta rekomendasi ke orang "B" yang suka anime dengan tema **slice-of-life** dan **comedy**, tentu saja rekomendasi dari orang "B" tidak pasti cocok dengan orang "A".

Oleh karena itu penggunaan sistem rekomendasi berbasis machine learning dapat memprediksi rekomendasi anime yang mungkin cocok dengan selera orang masing masing, meskipun tidak dapat memberi 100% kecocokan karena walaupun tema anime yang direkomendasikan sama, masih ada kemungkinan bahwa rekomendasi tersebut tidak cocok.

Selain itu platform untuk menonton anime seperti Crunchyroll atau Netflix sudah menggunakan sistem rekomendasi machine learning untuk memberikan rekomendasi film/anime yang mungkin cocok dengan usernya masing masing. Hal ini dilakukan untuk meningkatkan pengalaman pengguna dalam menemukan anime yang sesuai selera mereka. Teknik sistem rekomendasi machine learning yang dapat digunakan diantaranya adalah **Content-Based Filtering** dan **Collaborative Filtering**.

Teknik **Content-Based Filtering** dapat merekomendasikan suatu produk (dalam kasus ini anime) yang mungkin cocok dengan user berdasarkan fitur fitur yang terdapat pada produk tersebut. Contoh, orang "A" suka anime bernama **"Kizumonogatari I: Tekketsu-hen"** dengan genre atau tema **Action, Mystery, Vampire**. Teknik ini akan merekomendasikan anime dengan tema/genre yang sama.

Sedangkan teknik **Collaborative Filtering** dapat merekomendasikan suatu produk yang mungkin cocok dengan user berdasarkan produk yang disukai user itu di masa lalu dan produk yang disukai orang lain dengan selera yang sama. Contoh, orang "B" suka anime bernama **"Fullmetal Alchemist: Brotherhood"** dan orang "B" sudah memberikan rating sebesar 8.9/10. Selain itu orang "A" juga sudah memberikan rating untuk anime **"Fullmetal Alchemist: Brotherhood"** dan **"Kizumonogatari I: Tekketsu-hen"** dengan rating yang juga cukup tinggi. Maka dapat diasumsikan bahwa orang "B" berkemungkinan besar memiliki selera yang sama dengan orang "A". Lalu, sistem akan merekomendasikan anime yang sudah dirating oleh orang "A" yang belum ditonton oleh orang "B", yaitu **"Kizumonogatari I: Tekketsu-hen"**. 

## Business Understanding

### Problem Statement
- Bagaimana cara mendapatkan rekomendasi anime dengan genre yang mirip dengan anime yang disukai
- Bagaimana cara medapatkan rekomendasi anime yang mirip dengan anime yang pernah dirating

### Goals
- Mendapatkan rekomendasi anime berdasarkan genre atau tema dari anime yang disukai
- Mendapatkan rekomendasi anime berdasarkan anime yang pernah dirating

### Solution Approach
- Menggunakan teknik **Content-Based Filtering** untuk mendapatkan anime dengan genre yang mirip dengan anime yang disukai 
- Menggunakan teknik **Collaborative Filtering** untuk mendapatkan rekomendasi anime berdasarkan anime yang pernah dirating

## Data Understanding
Dataset untuk proyek ini dapat didownload melalui [link ini](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database).