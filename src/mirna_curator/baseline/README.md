# Baseline for miRNA GO curation

ePMC already has a lot of annotations for papers, including those we are interested in, could we use them to get an annotation, what would that look like?

Maybe we can decompose the flowchart into a set of requirements based on the nodes, and then encode the whole thing as a big nested if-statement


## GO:0035195
hasExperimentalEvidence: are there any annotations with ExperimentalMethods present?
functionalInteraction: Is there a luciferase assay in the experimental methods annotated? (misses other assays though)
effectOnEndogenousGeneExpression: ???

## GO:0035279
hasExperimentalEvidence: are there any annotations with ExperimentalMethods present?
functionalInteraction: Is there a luciferase assay in the experimental methods annotated? (misses other assays though)
effectOnEndogenousGeneExpression: ???
miRNAInhibitionChangesmRNALevel: ???

hasExperimentalEvidence: are there any annotations with ExperimentalMethods present?
miRNA-mRNABindingAssay: Look for luciferase or CrISPR/CAS9 in experimental methods annotations
effectOnEndogenousGeneExpression: ???
miRNAInhibitionChangesmRNALevel: ???

## GO:0035278
hasExperimentalEvidence: are there any annotations with ExperimentalMethods present?
functionalInteraction: Is there a luciferase assay in the experimental methods annotated? (misses other assays though)
effectOnEndogenousGeneExpression: ???
miRNAInhibitionChangesmRNALevel: ???

hasExperimentalEvidence: are there any annotations with ExperimentalMethods present?
miRNA-mRNABindingAssay: Look for luciferase or CrISPR/CAS9 in experimental methods annotations
effectOnEndogenousGeneExpression: ???
miRNAInhibitionChangesmRNALevel: ???



## Feasibility
It is probably possible to filter out papers without experimental methods using the annotations API, and maybe even possible to get papers with the right kinds of assays.

However, the annotations API at the moment doesn't really suit this problem because:

- One of the key things we're looking for (luciferase reporter assay) is not directly annotated as its own thing. E.g. https://europepmc.org/article/MED/23598417 where we have transfection(twice?), fluorescence and 'assay' all as separate things. We would have to do some inference to link these together which would be flaky

- There's a lot of noise. the above article has nearly 300 annotations, many duplicated and most not that useful

- If there is evidence for a particular method, there isn't usually anything to tell us what was measured. So trying to figure out the miRNA's effect on endogenous gene expression is impossible with these annotations.

- We have no way to figure out the context of the experiment. The linked example is using HepG2 cells in a disease context (I think, because of the annotation given to the paper). However, the annotations API can only tell us what cell lines were used.

- There is also a high change that we get false positives from this approach. e.g. https://europepmc.org/article/MED/36866529 which is a paper about the GO, yet has several of the annotations we would be looking for. It has experimental mentods when it _definitely shouldn't_ for example.
