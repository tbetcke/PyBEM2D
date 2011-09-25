from core.segments import Segment,Line,Arc,Ellipse,polygon
from core.mesh import Domain,Mesh
from core.bases import ElementToBasis, MeshToBasis, NodalLin, Legendre
from core.kernels import (LogSingleLayer, LogDoubleLayer, LogConjDoubleLayer,
AcousticSingleLayer, AcousticDoubleLayer, AcousticConjDoubleLayer,
AcousticCombined, Identity)
from core.assembly import Assembly
from core.quadrules import GaussQuadrature
from core.evaluation import Evaluator, evalDensity
from core.visualization import Visualizer, plotDensity




