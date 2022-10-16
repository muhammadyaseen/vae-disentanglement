# How to write thesis meeting notes

Usual length is 35-60 pages
 
### Ingredients of thesis

#### 1. Title
#### 2. Abstract
short overview of what the thesis is about... half to one page long

#### 3. Introducution
(is this where he mentioned the grandma example?)
- Needs to motivate the reader why should the reader read these pages.
3-5 pages.
- Intro is fleshed out version of the abstract .. it is at a high level
- Aim it at someone who is interested in the work but has no technical background.
- Always think of ur audience - other researchers from other areas
or readable by a smart bachelor student.
- Audience should get a general idea of what your're doing after intro

Should this include a flow of thesis para ? e.g we do X in chapter Y then Z in ch W and so on?

#### 4. Related work 

Think of it as a map of the world / map of our problem domain.

3-pages
- Know ur place in the world. 
- Know what has been done.. what's out there in the world.

trick: think of upsidedown christman tree
start with general stuff
building block of trees are paras e.g. first para can be about representations in NNs
becomes more specific as we go down
star at the top is ur method

should never bash other people

#### 5. Preliminaries (background)
3-5+ pages

 Notation
 Short intro of VAE etc.
 
#### 6. Theory
This part defines the problem
5+
- A formal definition of the problem e.g. what objective function are we minimizing
- e.g. ELBO and discuss parts of it

We can also include approaches that didn't quite work. e.g. Ladder VAE based and neural structure based.

premilnary result

#### 7. Algorithm - the solution
5+ 
Pseudo code of the pipeline
Active never passive

#### 8. Experiments (minimal interp) 
5-10+ pages

school trip to the zoo...guide chooses where the students go

come up with a bridge why we move from experiment to experiment
Those experiments that would convince the reader that the method does or doesn't work

e.g. first do sanity check - does it not find anything when it shouldn't
synth data + real data (often case study)

Why before how. First give me reason then tell me how u did it
What are the main qs that we want to have answered?

#### 9. Discussion 

(interpretation / implication / deeper thoughts) 
3-5+ pages
- most imp section - don't ignore this section
- showing off ur intelligence

Recap - one para summary of what we saw in experiments
All the derivate thoughts should be included in here.
Limitations and unmentioned strengths + improvements
For evey limitation come up with some idea on how can it be addressed.

#### 10. Conclusion
2 pages
Can be considered extended abstract in past tense

#### 11. Refrences

#### 12. Appendix
- additional experiments
- theorems / lemmas etc

### Which order to write it in?
1. Theory and Algorithm (Start with this - while writing theory make notes on what background / prelims to include)
2. Preliminaries (use macros `\newcommand \ensuremath \mathit` etc) 
3. Experiments (can be written in) parallel to Alg and Theory
4. Discussion
5. Intro
6. Abstract
7. Conclusion
8. Related work can be written whenever - it is kinda independent

- use `\mathit` for any function name / text etc.
- if text use `\emph`

when writing equations use inter punctuation
math / equations are just text.
dont use 'semi colon' to break text and eqn. use 'dot' / 'comman after the eqn.

- what tense POV is it written in  - active voice / we 
- technical format - 
- reference format

![[images/howtothesis.jpg]]

