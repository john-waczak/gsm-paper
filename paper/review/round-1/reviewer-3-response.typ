#import "response-template.typ": *

// Configure text colors for points, responses, and new text
#let (point, response, new) = configure(
  point-color: blue.darken(30%),
  response-color: black,
  new-color: green.darken(30%)
)

// Setup the rebuttal
#show: rebuttal.with(
  title: "Response to Reviewer #3 Comments",
  authors: [John Waczak and David Lary],
  date: "10-31-2024",
  paper-size: "us-letter",
)

We sincerely appreciate the time and effort that you dedicated to reviewing
our manuscript and are grateful for your insightful comments and suggested
improvements. Thank you!


#line(length: 100%)

#point[  
  The proposed Generative Simplex Mapping (GSM) model offers a fresh approach
  to non-linear endmember extraction and spectral unmixing in hyperspectral
  imagery, which is a significant contribution to the field. Highlighting the
  model’s flexibility to handle both linear and non-linear mixing, as well as
  its probabilistic nature, is commendable. However, it would be helpful if the
  authors more clearly articulated how the GSM directly advances the state of
  the art compared to other existing methods.
]<p1>

#response[
  Thank you for your kind comment. We have updated the introduction to explicitly
  include a list of the novel contributions made by our approach. The updated
  text now includes the following: 

  #quote[
    In summary, the key innovations introduced by the GSM are:
    - the GSM can model linear and nonlinear spectral mixing
    - the GSM does not assume the presence of pure pixels in the dataset
    - the probabilistic formulation of the GSM accounts for spectral variability
    - the simplex used for the latent space structure of the GSM is directly
      interpretable and forces abundances to satisfy both the abundance sum-to-one and
      abundance non-negativity constraints
    - the fiting procedure introduced for the GSM maintains non-negativity of endmember spectra.
  ]
]



#point[
  The description of the model, especially the non-linear mapping function and
  the (n−1)-simplex latent space, is interesting but requires more elaboration.
  Providing additional mathematical explanations or diagrams could help clarify
  the mechanics of the latent space and the transition between linear and
  non-linear regimes. Some readers may struggle with the abstract nature of the
  model description without further illustrative examples.
]<p2>

#response[
  Thank you for your comment. We have updated Section 2 to include additional descriptions for the model components. Specifically, the descriptions of Equations (1) and (2) have been extended with additional details for the variables. The description of the non-linear acitvation functions has also been updated and now reads as: 
  #quote[
    A visual representation of the non-linear activation functions in Equation
    4 is shown below in Figure 1 for a 2-component GSM. In this form, the first
    $N_v$ columns of $bold(W)$ correspond to endmember spectra, while the remaining
    columns account for additional non-linear effects. Equation 4 is specifically
    chosen so that no non-linear contributions are possible for pure spectra at the
    vertices of the simplex. At all other points, the output of $psi$ involves both
    linear and non-linear contributions. If only linear mixing is present, the GSM
    training algorithm should therefore drive $W_(d m)$ to $0$ for $m gt.eq N_v$.
  ]
  
   We have also added a new figure to illustrate the non-linear activation function for a 2-dimension GSM. 
]



#point[
  The comparison with three varieties of non-negative matrix factorization (NMF)
  on synthetic data is valuable. However, it would strengthen the evaluation if
  the authors included additional benchmark models, such as other widely used
  non-linear unmixing algorithms, to more comprehensively demonstrate GSM’s
  performance advantages. Are there specific scenarios where GSM is expected
  to significantly outperform standard methods, and if so, could these be
  highlighted?
]<p3>

#response[
  Our response...
]


#point[
  The synthetic data experiment and the real-world case study over a pond in North Texas are good choices for demonstrating the capabilities of GSM. Nonetheless, it would be beneficial to include more details on the real dataset, such as the resolution, spectral characteristics, and preprocessing steps, to help assess the generalizability of the method. Additionally, further validation on different types of real-world datasets with varying levels of complexity would strengthen the claims of the model’s robustness.
]<p4>

#response[
  add additional details for the dataset, \
  we inted to explore the model on more _real world_ datasets in subsequent work. For example, hyperspectral images from PACE, EnMAP, etc... could provide an interesting opportunity to perform additional experiments.
]

#point[
  The probabilistic treatment of spectral variability using a precision
  parameter is an interesting aspect of the model. However, the authors could
  provide more details on how this precision parameter is estimated, and how it
  affects the unmixing results. A deeper discussion of its impact on the overall
  performance of the GSM model, compared to deterministic approaches, would be
  valuable.
]<p5>

#response[
  Our response...
]

#point[
   While the model’s performance is highlighted, the computational cost
  of GSM relative to other models (e.g., NMF) is not fully discussed. Since
  hyperspectral data can be large and computationally demanding, it would be
  useful for the authors to provide an analysis of the algorithm's scalability
  and its computational requirements, especially for large datasets or real-time
  applications.
]<p6>

#response[
  I think we can address this by going into further detail about the size of the
  latent space and the number of latent space nodes
]

