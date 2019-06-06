# Transfer learning problems
Transfer learning problems can be catogoried into:<br>
### 1. Dataset Shift
*Ref: [Book Chapter](http://www.acad.bg/ebook/ml/The.MIT.Press.Dataset.Shift.in.Machine.Learning.Feb.2009.eBook-DDU.pdf) <br>*
This term refers to the problems where neither of marginal distribution and conditional distribution is same between Source and Target. <br>
***Ps(x) != Pt(x) and Ps(y|x) != Pt(y|x) <br>***
Few reserches are addressing this kind of problems, and most of them are based on instance reweighting methods. <br>
*Ref: [1](https://icml.cc/imls/conferences/2007/proceedings/papers/303.pdf); [2](http://sifaka.cs.uiuc.edu/czhai/pub/acl07.pdf); 
[3](https://www.andrewoarnold.com/arnolda-transfer-icdm-short.pdf); [4](https://dl.acm.org/citation.cfm?id=1557130).<br>*

### 2. Domain Adaption
This term is used where the marginal distributions are different, but we assume conditional distribution is the same. <br>
Two methods: 1) Instance weighting, learning the instance weights ***w(x)***; <br>
e.g. KMM, [KL-based KLIEP](https://papers.nips.cc/paper/3248-direct-importance-estimation-with-model-selection-and-its-application-to-covariate-shift-adaptation.pdf), TrAdaBoost.<br>
Only works better when distribution descrepency is not large. <br>
2) Feature expression: leaning the mapping function of feature space phi(X). Such that the Dist( Ps(phi(x)), Pt(phi(x) ) is minimized. <br>

### 3. Multi-task Learning
This is another mainstream of transfer learning. The conditional distributions are different between domains. (Is the marginal distributions equal?)<br>
The difference between DA and Multi-task: DA focuses on the performance on only the target domain. Multi-task requires labeled target domain.<br>

