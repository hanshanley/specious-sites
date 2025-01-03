# Specious Sites: Tracking the Spread and Sway of Spurious News Stories at Scale

Misinformation, propaganda, and outright lies proliferate on the web, with some narratives having dangerous real-world consequences on public health, elections, and individual safety. However, despite the impact of misinformation, the research community largely lacks automated and programmatic approaches for tracking news narratives across online platforms. In this work, utilizing daily scrapes of 1,334 unreliable news websites, the large-language model MPNet, and DP-Means clustering, we introduce a system to automatically identify and track the narratives spread within online ecosystems. Identifying 52,036 narratives on these 1,334 websites, we describe the most prevalent narratives spread in 2022 and identify the most influential websites that originate and amplify narratives. Finally, we show how our system can be utilized to detect new narratives originating from unreliable news websites and to aid fact-checkers in more quickly addressing misinformation.

https://www.computer.org/csdl/proceedings-article/sp/2024/313000a180/1V28Z4xTqTK

## Dp-Means Clustering 
The folder `dpmeans_clustering` contains code that performs an optimized version of DP-Means Clustering using cosine similarity rather than Euclidean distance. 
 
## Website List
This GitHub repository contains the websites utilized to document the popularity and influence of different unreliable and biased websites. This dataset consists of 1,334 unreliable and biased websites. 

For additional details about the collection method and analysis of these websites, see our paper/analysis here: https://www.hanshanley.com/files/Specious_Sites.pdf

## Citing the paper
If our lists of sites are useful for your own research, you can cite us with the following BibTex entry:
```
@INPROCEEDINGS {10646651,
author = { Hanley, Hans W. A. and Kumar, Deepak and Durumeric, Zakir },
booktitle = { 2024 IEEE Symposium on Security and Privacy (SP) },
title = {{ Specious Sites: Tracking the Spread and Sway of Spurious News Stories at Scale }},
year = {2024},
volume = {},
ISSN = {},
pages = {1609-1627},
keywords = {Privacy;Codes;Voting;Ecosystems;Safety;Security;Fake news},
doi = {10.1109/SP54263.2024.00171},
url = {https://doi.ieeecomputersociety.org/10.1109/SP54263.2024.00171},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month =May}
```
## License and Copyright

Copyright 2024 The Board of Trustees of The Leland Stanford Junior University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

