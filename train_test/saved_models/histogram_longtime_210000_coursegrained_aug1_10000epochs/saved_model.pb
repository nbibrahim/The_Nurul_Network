��	
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628��
Z
ConstConst*
_output_shapes

:*
dtype0*
valueB*  �?
\
Const_1Const*
_output_shapes

:*
dtype0*
valueB*    
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
�
Adam/v/dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_14/bias
y
(Adam/v/dense_14/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_14/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_14/bias
y
(Adam/m/dense_14/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_14/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_14/kernel
�
*Adam/v/dense_14/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_14/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_14/kernel
�
*Adam/m/dense_14/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_14/kernel*
_output_shapes

:*
dtype0
�
Adam/v/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_13/bias
y
(Adam/v/dense_13/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_13/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_13/bias
y
(Adam/m/dense_13/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_13/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/v/dense_13/kernel
�
*Adam/v/dense_13/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_13/kernel*
_output_shapes

:
*
dtype0
�
Adam/m/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/m/dense_13/kernel
�
*Adam/m/dense_13/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_13/kernel*
_output_shapes

:
*
dtype0
�
Adam/v/dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/v/dense_12/bias
y
(Adam/v/dense_12/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_12/bias*
_output_shapes
:
*
dtype0
�
Adam/m/dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/m/dense_12/bias
y
(Adam/m/dense_12/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_12/bias*
_output_shapes
:
*
dtype0
�
Adam/v/dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/v/dense_12/kernel
�
*Adam/v/dense_12/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_12/kernel*
_output_shapes

:
*
dtype0
�
Adam/m/dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/m/dense_12/kernel
�
*Adam/m/dense_12/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_12/kernel*
_output_shapes

:
*
dtype0
�
Adam/v/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_11/bias
y
(Adam/v/dense_11/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_11/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_11/bias
y
(Adam/m/dense_11/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_11/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_11/kernel
�
*Adam/v/dense_11/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_11/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_11/kernel
�
*Adam/m/dense_11/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_11/kernel*
_output_shapes

:*
dtype0
�
Adam/v/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_10/bias
y
(Adam/v/dense_10/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_10/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_10/bias
y
(Adam/m/dense_10/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_10/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_10/kernel
�
*Adam/v/dense_10/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_10/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_10/kernel
�
*Adam/m/dense_10/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_10/kernel*
_output_shapes

:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:*
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:
*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:
*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:
*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:*
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:*
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:*
dtype0
~
normalization_2/countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *&
shared_namenormalization_2/count
w
)normalization_2/count/Read/ReadVariableOpReadVariableOpnormalization_2/count*
_output_shapes
: *
dtype0	
�
normalization_2/varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namenormalization_2/variance
�
,normalization_2/variance/Read/ReadVariableOpReadVariableOpnormalization_2/variance*
_output_shapes
:*
dtype0
�
normalization_2/meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namenormalization_2/mean
y
(normalization_2/mean/Read/ReadVariableOpReadVariableOpnormalization_2/mean*
_output_shapes
:*
dtype0
�
%serving_default_normalization_2_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall%serving_default_normalization_2_inputConst_1Constdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*8
config_proto(&

CPU

GPU2*0J

  �E8� *.
f)R'
%__inference_signature_wrapper_4874230

NoOpNoOp
�G
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*�F
value�FB�F B�F
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias*
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator* 
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias*
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias*
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias*
b
0
1
2
3
 4
.5
/6
67
78
>9
?10
F11
G12*
J
0
 1
.2
/3
64
75
>6
?7
F8
G9*
* 
�
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Mtrace_0
Ntrace_1* 

Otrace_0
Ptrace_1* 
 
Q	capture_0
R	capture_1* 
�
S
_variables
T_iterations
U_learning_rate
V_index_dict
W
_momentums
X_velocities
Y_update_step_xla*

Zserving_default* 
* 
* 
* 
* 
* 
b\
VARIABLE_VALUEnormalization_2/mean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEnormalization_2/variance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEnormalization_2/count5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
* 
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

`trace_0* 

atrace_0* 
_Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_10/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses* 

gtrace_0
htrace_1* 

itrace_0
jtrace_1* 
* 

.0
/1*

.0
/1*
* 
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

ptrace_0* 

qtrace_0* 
_Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_11/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

60
71*

60
71*
* 
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

wtrace_0* 

xtrace_0* 
_Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_12/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

>0
?1*

>0
?1*
* 
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

~trace_0* 

trace_0* 
_Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_13/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

F0
G1*

F0
G1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_14/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1
2*
5
0
1
2
3
4
5
6*

�0*
* 
* 
 
Q	capture_0
R	capture_1* 
 
Q	capture_0
R	capture_1* 
 
Q	capture_0
R	capture_1* 
 
Q	capture_0
R	capture_1* 
* 
* 
�
T0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
T
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9*
T
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9*
* 
 
Q	capture_0
R	capture_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
a[
VARIABLE_VALUEAdam/m/dense_10/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_10/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_10/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_10/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_11/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_11/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_11/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_11/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_12/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_12/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_12/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_12/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_13/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_13/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_13/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_13/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_14/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_14/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_14/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_14/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamenormalization_2/meannormalization_2/variancenormalization_2/countdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/bias	iterationlearning_rateAdam/m/dense_10/kernelAdam/v/dense_10/kernelAdam/m/dense_10/biasAdam/v/dense_10/biasAdam/m/dense_11/kernelAdam/v/dense_11/kernelAdam/m/dense_11/biasAdam/v/dense_11/biasAdam/m/dense_12/kernelAdam/v/dense_12/kernelAdam/m/dense_12/biasAdam/v/dense_12/biasAdam/m/dense_13/kernelAdam/v/dense_13/kernelAdam/m/dense_13/biasAdam/v/dense_13/biasAdam/m/dense_14/kernelAdam/v/dense_14/kernelAdam/m/dense_14/biasAdam/v/dense_14/biastotalcountConst_2*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  �E8� *)
f$R"
 __inference__traced_save_4874602
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamenormalization_2/meannormalization_2/variancenormalization_2/countdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/bias	iterationlearning_rateAdam/m/dense_10/kernelAdam/v/dense_10/kernelAdam/m/dense_10/biasAdam/v/dense_10/biasAdam/m/dense_11/kernelAdam/v/dense_11/kernelAdam/m/dense_11/biasAdam/v/dense_11/biasAdam/m/dense_12/kernelAdam/v/dense_12/kernelAdam/m/dense_12/biasAdam/v/dense_12/biasAdam/m/dense_13/kernelAdam/v/dense_13/kernelAdam/m/dense_13/biasAdam/v/dense_13/biasAdam/m/dense_14/kernelAdam/v/dense_14/kernelAdam/m/dense_14/biasAdam/v/dense_14/biastotalcount*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  �E8� *,
f'R%
#__inference__traced_restore_4874722��
��
�
#__inference__traced_restore_4874722
file_prefix3
%assignvariableop_normalization_2_mean:9
+assignvariableop_1_normalization_2_variance:2
(assignvariableop_2_normalization_2_count:	 4
"assignvariableop_3_dense_10_kernel:.
 assignvariableop_4_dense_10_bias:4
"assignvariableop_5_dense_11_kernel:.
 assignvariableop_6_dense_11_bias:4
"assignvariableop_7_dense_12_kernel:
.
 assignvariableop_8_dense_12_bias:
4
"assignvariableop_9_dense_13_kernel:
/
!assignvariableop_10_dense_13_bias:5
#assignvariableop_11_dense_14_kernel:/
!assignvariableop_12_dense_14_bias:'
assignvariableop_13_iteration:	 +
!assignvariableop_14_learning_rate: <
*assignvariableop_15_adam_m_dense_10_kernel:<
*assignvariableop_16_adam_v_dense_10_kernel:6
(assignvariableop_17_adam_m_dense_10_bias:6
(assignvariableop_18_adam_v_dense_10_bias:<
*assignvariableop_19_adam_m_dense_11_kernel:<
*assignvariableop_20_adam_v_dense_11_kernel:6
(assignvariableop_21_adam_m_dense_11_bias:6
(assignvariableop_22_adam_v_dense_11_bias:<
*assignvariableop_23_adam_m_dense_12_kernel:
<
*assignvariableop_24_adam_v_dense_12_kernel:
6
(assignvariableop_25_adam_m_dense_12_bias:
6
(assignvariableop_26_adam_v_dense_12_bias:
<
*assignvariableop_27_adam_m_dense_13_kernel:
<
*assignvariableop_28_adam_v_dense_13_kernel:
6
(assignvariableop_29_adam_m_dense_13_bias:6
(assignvariableop_30_adam_v_dense_13_bias:<
*assignvariableop_31_adam_m_dense_14_kernel:<
*assignvariableop_32_adam_v_dense_14_kernel:6
(assignvariableop_33_adam_m_dense_14_bias:6
(assignvariableop_34_adam_v_dense_14_bias:#
assignvariableop_35_total: #
assignvariableop_36_count: 
identity_38��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*�
value�B�&B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp%assignvariableop_normalization_2_meanIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp+assignvariableop_1_normalization_2_varianceIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp(assignvariableop_2_normalization_2_countIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_10_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_10_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_11_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_11_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_12_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_12_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_13_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense_13_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_14_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense_14_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_iterationIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp!assignvariableop_14_learning_rateIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_m_dense_10_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_v_dense_10_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_m_dense_10_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_v_dense_10_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_m_dense_11_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_v_dense_11_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_m_dense_11_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_v_dense_11_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_m_dense_12_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_v_dense_12_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_m_dense_12_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_v_dense_12_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_m_dense_13_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_v_dense_13_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_m_dense_13_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_v_dense_13_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_m_dense_14_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_v_dense_14_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_m_dense_14_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_v_dense_14_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpassignvariableop_35_totalIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpassignvariableop_36_countIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_38IdentityIdentity_37:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_38Identity_38:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%%!

_user_specified_namecount:%$!

_user_specified_nametotal:4#0
.
_user_specified_nameAdam/v/dense_14/bias:4"0
.
_user_specified_nameAdam/m/dense_14/bias:6!2
0
_user_specified_nameAdam/v/dense_14/kernel:6 2
0
_user_specified_nameAdam/m/dense_14/kernel:40
.
_user_specified_nameAdam/v/dense_13/bias:40
.
_user_specified_nameAdam/m/dense_13/bias:62
0
_user_specified_nameAdam/v/dense_13/kernel:62
0
_user_specified_nameAdam/m/dense_13/kernel:40
.
_user_specified_nameAdam/v/dense_12/bias:40
.
_user_specified_nameAdam/m/dense_12/bias:62
0
_user_specified_nameAdam/v/dense_12/kernel:62
0
_user_specified_nameAdam/m/dense_12/kernel:40
.
_user_specified_nameAdam/v/dense_11/bias:40
.
_user_specified_nameAdam/m/dense_11/bias:62
0
_user_specified_nameAdam/v/dense_11/kernel:62
0
_user_specified_nameAdam/m/dense_11/kernel:40
.
_user_specified_nameAdam/v/dense_10/bias:40
.
_user_specified_nameAdam/m/dense_10/bias:62
0
_user_specified_nameAdam/v/dense_10/kernel:62
0
_user_specified_nameAdam/m/dense_10/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namedense_14/bias:/+
)
_user_specified_namedense_14/kernel:-)
'
_user_specified_namedense_13/bias:/
+
)
_user_specified_namedense_13/kernel:-	)
'
_user_specified_namedense_12/bias:/+
)
_user_specified_namedense_12/kernel:-)
'
_user_specified_namedense_11/bias:/+
)
_user_specified_namedense_11/kernel:-)
'
_user_specified_namedense_10/bias:/+
)
_user_specified_namedense_10/kernel:51
/
_user_specified_namenormalization_2/count:84
2
_user_specified_namenormalization_2/variance:40
.
_user_specified_namenormalization_2/mean:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
*__inference_dense_10_layer_call_fn_4874239

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  �E8� *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_4873959o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4874235:'#
!
_user_specified_name	4874233:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_10_layer_call_and_return_conditional_losses_4874250

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_11_layer_call_and_return_conditional_losses_4874297

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_11_layer_call_and_return_conditional_losses_4873988

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
E__inference_dense_14_layer_call_and_return_conditional_losses_4874356

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

e
F__inference_dropout_2_layer_call_and_return_conditional_losses_4873976

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_12_layer_call_fn_4874306

inputs
unknown:

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  �E8� *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_4874004o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4874302:'#
!
_user_specified_name	4874300:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
G
+__inference_dropout_2_layer_call_fn_4874260

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  �E8� *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_4874062`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

e
F__inference_dropout_2_layer_call_and_return_conditional_losses_4874272

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_4874230
normalization_2_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:

	unknown_6:

	unknown_7:

	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*8
config_proto(&

CPU

GPU2*0J

  �E8� *+
f&R$
"__inference__wrapped_model_4873938o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������::: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4874226:'#
!
_user_specified_name	4874224:'
#
!
_user_specified_name	4874222:'	#
!
_user_specified_name	4874220:'#
!
_user_specified_name	4874218:'#
!
_user_specified_name	4874216:'#
!
_user_specified_name	4874214:'#
!
_user_specified_name	4874212:'#
!
_user_specified_name	4874210:'#
!
_user_specified_name	4874208:$ 

_output_shapes

::$ 

_output_shapes

::^ Z
'
_output_shapes
:���������
/
_user_specified_namenormalization_2_input
Ě
�!
 __inference__traced_save_4874602
file_prefix9
+read_disablecopyonread_normalization_2_mean:?
1read_1_disablecopyonread_normalization_2_variance:8
.read_2_disablecopyonread_normalization_2_count:	 :
(read_3_disablecopyonread_dense_10_kernel:4
&read_4_disablecopyonread_dense_10_bias::
(read_5_disablecopyonread_dense_11_kernel:4
&read_6_disablecopyonread_dense_11_bias::
(read_7_disablecopyonread_dense_12_kernel:
4
&read_8_disablecopyonread_dense_12_bias:
:
(read_9_disablecopyonread_dense_13_kernel:
5
'read_10_disablecopyonread_dense_13_bias:;
)read_11_disablecopyonread_dense_14_kernel:5
'read_12_disablecopyonread_dense_14_bias:-
#read_13_disablecopyonread_iteration:	 1
'read_14_disablecopyonread_learning_rate: B
0read_15_disablecopyonread_adam_m_dense_10_kernel:B
0read_16_disablecopyonread_adam_v_dense_10_kernel:<
.read_17_disablecopyonread_adam_m_dense_10_bias:<
.read_18_disablecopyonread_adam_v_dense_10_bias:B
0read_19_disablecopyonread_adam_m_dense_11_kernel:B
0read_20_disablecopyonread_adam_v_dense_11_kernel:<
.read_21_disablecopyonread_adam_m_dense_11_bias:<
.read_22_disablecopyonread_adam_v_dense_11_bias:B
0read_23_disablecopyonread_adam_m_dense_12_kernel:
B
0read_24_disablecopyonread_adam_v_dense_12_kernel:
<
.read_25_disablecopyonread_adam_m_dense_12_bias:
<
.read_26_disablecopyonread_adam_v_dense_12_bias:
B
0read_27_disablecopyonread_adam_m_dense_13_kernel:
B
0read_28_disablecopyonread_adam_v_dense_13_kernel:
<
.read_29_disablecopyonread_adam_m_dense_13_bias:<
.read_30_disablecopyonread_adam_v_dense_13_bias:B
0read_31_disablecopyonread_adam_m_dense_14_kernel:B
0read_32_disablecopyonread_adam_v_dense_14_kernel:<
.read_33_disablecopyonread_adam_m_dense_14_bias:<
.read_34_disablecopyonread_adam_v_dense_14_bias:)
read_35_disablecopyonread_total: )
read_36_disablecopyonread_count: 
savev2_const_2
identity_75��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: }
Read/DisableCopyOnReadDisableCopyOnRead+read_disablecopyonread_normalization_2_mean"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp+read_disablecopyonread_normalization_2_mean^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0e
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:]

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_1/DisableCopyOnReadDisableCopyOnRead1read_1_disablecopyonread_normalization_2_variance"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp1read_1_disablecopyonread_normalization_2_variance^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_2/DisableCopyOnReadDisableCopyOnRead.read_2_disablecopyonread_normalization_2_count"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp.read_2_disablecopyonread_normalization_2_count^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	e

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: [

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_3/DisableCopyOnReadDisableCopyOnRead(read_3_disablecopyonread_dense_10_kernel"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp(read_3_disablecopyonread_dense_10_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_4/DisableCopyOnReadDisableCopyOnRead&read_4_disablecopyonread_dense_10_bias"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp&read_4_disablecopyonread_dense_10_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_5/DisableCopyOnReadDisableCopyOnRead(read_5_disablecopyonread_dense_11_kernel"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp(read_5_disablecopyonread_dense_11_kernel^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_6/DisableCopyOnReadDisableCopyOnRead&read_6_disablecopyonread_dense_11_bias"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp&read_6_disablecopyonread_dense_11_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_7/DisableCopyOnReadDisableCopyOnRead(read_7_disablecopyonread_dense_12_kernel"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp(read_7_disablecopyonread_dense_12_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0n
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

:
z
Read_8/DisableCopyOnReadDisableCopyOnRead&read_8_disablecopyonread_dense_12_bias"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp&read_8_disablecopyonread_dense_12_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:
|
Read_9/DisableCopyOnReadDisableCopyOnRead(read_9_disablecopyonread_dense_13_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp(read_9_disablecopyonread_dense_13_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

:
|
Read_10/DisableCopyOnReadDisableCopyOnRead'read_10_disablecopyonread_dense_13_bias"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp'read_10_disablecopyonread_dense_13_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_11/DisableCopyOnReadDisableCopyOnRead)read_11_disablecopyonread_dense_14_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp)read_11_disablecopyonread_dense_14_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:|
Read_12/DisableCopyOnReadDisableCopyOnRead'read_12_disablecopyonread_dense_14_bias"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp'read_12_disablecopyonread_dense_14_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_13/DisableCopyOnReadDisableCopyOnRead#read_13_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp#read_13_disablecopyonread_iteration^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_14/DisableCopyOnReadDisableCopyOnRead'read_14_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp'read_14_disablecopyonread_learning_rate^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_15/DisableCopyOnReadDisableCopyOnRead0read_15_disablecopyonread_adam_m_dense_10_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp0read_15_disablecopyonread_adam_m_dense_10_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_16/DisableCopyOnReadDisableCopyOnRead0read_16_disablecopyonread_adam_v_dense_10_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp0read_16_disablecopyonread_adam_v_dense_10_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_17/DisableCopyOnReadDisableCopyOnRead.read_17_disablecopyonread_adam_m_dense_10_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp.read_17_disablecopyonread_adam_m_dense_10_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_18/DisableCopyOnReadDisableCopyOnRead.read_18_disablecopyonread_adam_v_dense_10_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp.read_18_disablecopyonread_adam_v_dense_10_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnRead0read_19_disablecopyonread_adam_m_dense_11_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp0read_19_disablecopyonread_adam_m_dense_11_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_20/DisableCopyOnReadDisableCopyOnRead0read_20_disablecopyonread_adam_v_dense_11_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp0read_20_disablecopyonread_adam_v_dense_11_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_21/DisableCopyOnReadDisableCopyOnRead.read_21_disablecopyonread_adam_m_dense_11_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp.read_21_disablecopyonread_adam_m_dense_11_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_22/DisableCopyOnReadDisableCopyOnRead.read_22_disablecopyonread_adam_v_dense_11_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp.read_22_disablecopyonread_adam_v_dense_11_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_adam_m_dense_12_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_adam_m_dense_12_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0o
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_24/DisableCopyOnReadDisableCopyOnRead0read_24_disablecopyonread_adam_v_dense_12_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp0read_24_disablecopyonread_adam_v_dense_12_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0o
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_25/DisableCopyOnReadDisableCopyOnRead.read_25_disablecopyonread_adam_m_dense_12_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp.read_25_disablecopyonread_adam_m_dense_12_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_26/DisableCopyOnReadDisableCopyOnRead.read_26_disablecopyonread_adam_v_dense_12_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp.read_26_disablecopyonread_adam_v_dense_12_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_27/DisableCopyOnReadDisableCopyOnRead0read_27_disablecopyonread_adam_m_dense_13_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp0read_27_disablecopyonread_adam_m_dense_13_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0o
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_28/DisableCopyOnReadDisableCopyOnRead0read_28_disablecopyonread_adam_v_dense_13_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp0read_28_disablecopyonread_adam_v_dense_13_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0o
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_29/DisableCopyOnReadDisableCopyOnRead.read_29_disablecopyonread_adam_m_dense_13_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp.read_29_disablecopyonread_adam_m_dense_13_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_30/DisableCopyOnReadDisableCopyOnRead.read_30_disablecopyonread_adam_v_dense_13_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp.read_30_disablecopyonread_adam_v_dense_13_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_31/DisableCopyOnReadDisableCopyOnRead0read_31_disablecopyonread_adam_m_dense_14_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp0read_31_disablecopyonread_adam_m_dense_14_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_32/DisableCopyOnReadDisableCopyOnRead0read_32_disablecopyonread_adam_v_dense_14_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp0read_32_disablecopyonread_adam_v_dense_14_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_33/DisableCopyOnReadDisableCopyOnRead.read_33_disablecopyonread_adam_m_dense_14_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp.read_33_disablecopyonread_adam_m_dense_14_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_34/DisableCopyOnReadDisableCopyOnRead.read_34_disablecopyonread_adam_v_dense_14_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp.read_34_disablecopyonread_adam_v_dense_14_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_35/DisableCopyOnReadDisableCopyOnReadread_35_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOpread_35_disablecopyonread_total^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_36/DisableCopyOnReadDisableCopyOnReadread_36_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOpread_36_disablecopyonread_count^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*�
value�B�&B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0savev2_const_2"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *4
dtypes*
(2&		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_74Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_75IdentityIdentity_74:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_75Identity_75:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:?&;

_output_shapes
: 
!
_user_specified_name	Const_2:%%!

_user_specified_namecount:%$!

_user_specified_nametotal:4#0
.
_user_specified_nameAdam/v/dense_14/bias:4"0
.
_user_specified_nameAdam/m/dense_14/bias:6!2
0
_user_specified_nameAdam/v/dense_14/kernel:6 2
0
_user_specified_nameAdam/m/dense_14/kernel:40
.
_user_specified_nameAdam/v/dense_13/bias:40
.
_user_specified_nameAdam/m/dense_13/bias:62
0
_user_specified_nameAdam/v/dense_13/kernel:62
0
_user_specified_nameAdam/m/dense_13/kernel:40
.
_user_specified_nameAdam/v/dense_12/bias:40
.
_user_specified_nameAdam/m/dense_12/bias:62
0
_user_specified_nameAdam/v/dense_12/kernel:62
0
_user_specified_nameAdam/m/dense_12/kernel:40
.
_user_specified_nameAdam/v/dense_11/bias:40
.
_user_specified_nameAdam/m/dense_11/bias:62
0
_user_specified_nameAdam/v/dense_11/kernel:62
0
_user_specified_nameAdam/m/dense_11/kernel:40
.
_user_specified_nameAdam/v/dense_10/bias:40
.
_user_specified_nameAdam/m/dense_10/bias:62
0
_user_specified_nameAdam/v/dense_10/kernel:62
0
_user_specified_nameAdam/m/dense_10/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namedense_14/bias:/+
)
_user_specified_namedense_14/kernel:-)
'
_user_specified_namedense_13/bias:/
+
)
_user_specified_namedense_13/kernel:-	)
'
_user_specified_namedense_12/bias:/+
)
_user_specified_namedense_12/kernel:-)
'
_user_specified_namedense_11/bias:/+
)
_user_specified_namedense_11/kernel:-)
'
_user_specified_namedense_10/bias:/+
)
_user_specified_namedense_10/kernel:51
/
_user_specified_namenormalization_2/count:84
2
_user_specified_namenormalization_2/variance:40
.
_user_specified_namenormalization_2/mean:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
d
+__inference_dropout_2_layer_call_fn_4874255

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  �E8� *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_4873976o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_11_layer_call_fn_4874286

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  �E8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_4873988o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4874282:'#
!
_user_specified_name	4874280:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_13_layer_call_and_return_conditional_losses_4874020

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_4874062

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_13_layer_call_and_return_conditional_losses_4874337

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
E__inference_dense_12_layer_call_and_return_conditional_losses_4874317

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������
W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_14_layer_call_fn_4874346

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  �E8� *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_4874035o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4874342:'#
!
_user_specified_name	4874340:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�*
�
I__inference_sequential_2_layer_call_and_return_conditional_losses_4874042
normalization_2_input
normalization_2_sub_y
normalization_2_sqrt_x"
dense_10_4873960:
dense_10_4873962:"
dense_11_4873989:
dense_11_4873991:"
dense_12_4874005:

dense_12_4874007:
"
dense_13_4874021:

dense_13_4874023:"
dense_14_4874036:
dense_14_4874038:
identity�� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCallt
normalization_2/CastCastnormalization_2_input*

DstT0*

SrcT0*'
_output_shapes
:���������}
normalization_2/subSubnormalization_2/Cast:y:0normalization_2_sub_y*
T0*'
_output_shapes
:���������]
normalization_2/SqrtSqrtnormalization_2_sqrt_x*
T0*
_output_shapes

:^
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:�
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:����������
 dense_10/StatefulPartitionedCallStatefulPartitionedCallnormalization_2/truediv:z:0dense_10_4873960dense_10_4873962*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  �E8� *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_4873959�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  �E8� *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_4873976�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_11_4873989dense_11_4873991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  �E8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_4873988�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_4874005dense_12_4874007*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  �E8� *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_4874004�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_4874021dense_13_4874023*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  �E8� *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_4874020�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_4874036dense_14_4874038*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  �E8� *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_4874035x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������::: : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:'#
!
_user_specified_name	4874038:'#
!
_user_specified_name	4874036:'
#
!
_user_specified_name	4874023:'	#
!
_user_specified_name	4874021:'#
!
_user_specified_name	4874007:'#
!
_user_specified_name	4874005:'#
!
_user_specified_name	4873991:'#
!
_user_specified_name	4873989:'#
!
_user_specified_name	4873962:'#
!
_user_specified_name	4873960:$ 

_output_shapes

::$ 

_output_shapes

::^ Z
'
_output_shapes
:���������
/
_user_specified_namenormalization_2_input
�)
�
I__inference_sequential_2_layer_call_and_return_conditional_losses_4874085
normalization_2_input
normalization_2_sub_y
normalization_2_sqrt_x"
dense_10_4874053:
dense_10_4874055:"
dense_11_4874064:
dense_11_4874066:"
dense_12_4874069:

dense_12_4874071:
"
dense_13_4874074:

dense_13_4874076:"
dense_14_4874079:
dense_14_4874081:
identity�� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCallt
normalization_2/CastCastnormalization_2_input*

DstT0*

SrcT0*'
_output_shapes
:���������}
normalization_2/subSubnormalization_2/Cast:y:0normalization_2_sub_y*
T0*'
_output_shapes
:���������]
normalization_2/SqrtSqrtnormalization_2_sqrt_x*
T0*
_output_shapes

:^
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:�
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:����������
 dense_10/StatefulPartitionedCallStatefulPartitionedCallnormalization_2/truediv:z:0dense_10_4874053dense_10_4874055*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  �E8� *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_4873959�
dropout_2/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  �E8� *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_4874062�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_11_4874064dense_11_4874066*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  �E8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_4873988�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_4874069dense_12_4874071*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  �E8� *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_4874004�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_4874074dense_13_4874076*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  �E8� *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_4874020�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_4874079dense_14_4874081*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  �E8� *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_4874035x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������::: : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:'#
!
_user_specified_name	4874081:'#
!
_user_specified_name	4874079:'
#
!
_user_specified_name	4874076:'	#
!
_user_specified_name	4874074:'#
!
_user_specified_name	4874071:'#
!
_user_specified_name	4874069:'#
!
_user_specified_name	4874066:'#
!
_user_specified_name	4874064:'#
!
_user_specified_name	4874055:'#
!
_user_specified_name	4874053:$ 

_output_shapes

::$ 

_output_shapes

::^ Z
'
_output_shapes
:���������
/
_user_specified_namenormalization_2_input
�

�
E__inference_dense_10_layer_call_and_return_conditional_losses_4873959

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_12_layer_call_and_return_conditional_losses_4874004

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������
W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_sequential_2_layer_call_fn_4874114
normalization_2_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:

	unknown_6:

	unknown_7:

	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*8
config_proto(&

CPU

GPU2*0J

  �E8� *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_4874042o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������::: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4874110:'#
!
_user_specified_name	4874108:'
#
!
_user_specified_name	4874106:'	#
!
_user_specified_name	4874104:'#
!
_user_specified_name	4874102:'#
!
_user_specified_name	4874100:'#
!
_user_specified_name	4874098:'#
!
_user_specified_name	4874096:'#
!
_user_specified_name	4874094:'#
!
_user_specified_name	4874092:$ 

_output_shapes

::$ 

_output_shapes

::^ Z
'
_output_shapes
:���������
/
_user_specified_namenormalization_2_input
�
�
.__inference_sequential_2_layer_call_fn_4874143
normalization_2_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:

	unknown_6:

	unknown_7:

	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*8
config_proto(&

CPU

GPU2*0J

  �E8� *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_4874085o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������::: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4874139:'#
!
_user_specified_name	4874137:'
#
!
_user_specified_name	4874135:'	#
!
_user_specified_name	4874133:'#
!
_user_specified_name	4874131:'#
!
_user_specified_name	4874129:'#
!
_user_specified_name	4874127:'#
!
_user_specified_name	4874125:'#
!
_user_specified_name	4874123:'#
!
_user_specified_name	4874121:$ 

_output_shapes

::$ 

_output_shapes

::^ Z
'
_output_shapes
:���������
/
_user_specified_namenormalization_2_input
�
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_4874277

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
E__inference_dense_14_layer_call_and_return_conditional_losses_4874035

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_13_layer_call_fn_4874326

inputs
unknown:

	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  �E8� *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_4874020o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4874322:'#
!
_user_specified_name	4874320:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�C
�

"__inference__wrapped_model_4873938
normalization_2_input&
"sequential_2_normalization_2_sub_y'
#sequential_2_normalization_2_sqrt_xF
4sequential_2_dense_10_matmul_readvariableop_resource:C
5sequential_2_dense_10_biasadd_readvariableop_resource:F
4sequential_2_dense_11_matmul_readvariableop_resource:C
5sequential_2_dense_11_biasadd_readvariableop_resource:F
4sequential_2_dense_12_matmul_readvariableop_resource:
C
5sequential_2_dense_12_biasadd_readvariableop_resource:
F
4sequential_2_dense_13_matmul_readvariableop_resource:
C
5sequential_2_dense_13_biasadd_readvariableop_resource:F
4sequential_2_dense_14_matmul_readvariableop_resource:C
5sequential_2_dense_14_biasadd_readvariableop_resource:
identity��,sequential_2/dense_10/BiasAdd/ReadVariableOp�+sequential_2/dense_10/MatMul/ReadVariableOp�,sequential_2/dense_11/BiasAdd/ReadVariableOp�+sequential_2/dense_11/MatMul/ReadVariableOp�,sequential_2/dense_12/BiasAdd/ReadVariableOp�+sequential_2/dense_12/MatMul/ReadVariableOp�,sequential_2/dense_13/BiasAdd/ReadVariableOp�+sequential_2/dense_13/MatMul/ReadVariableOp�,sequential_2/dense_14/BiasAdd/ReadVariableOp�+sequential_2/dense_14/MatMul/ReadVariableOp�
!sequential_2/normalization_2/CastCastnormalization_2_input*

DstT0*

SrcT0*'
_output_shapes
:����������
 sequential_2/normalization_2/subSub%sequential_2/normalization_2/Cast:y:0"sequential_2_normalization_2_sub_y*
T0*'
_output_shapes
:���������w
!sequential_2/normalization_2/SqrtSqrt#sequential_2_normalization_2_sqrt_x*
T0*
_output_shapes

:k
&sequential_2/normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
$sequential_2/normalization_2/MaximumMaximum%sequential_2/normalization_2/Sqrt:y:0/sequential_2/normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:�
$sequential_2/normalization_2/truedivRealDiv$sequential_2/normalization_2/sub:z:0(sequential_2/normalization_2/Maximum:z:0*
T0*'
_output_shapes
:����������
+sequential_2/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential_2/dense_10/MatMulMatMul(sequential_2/normalization_2/truediv:z:03sequential_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,sequential_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_2/dense_10/BiasAddBiasAdd&sequential_2/dense_10/MatMul:product:04sequential_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
sequential_2/dense_10/TanhTanh&sequential_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������}
sequential_2/dropout_2/IdentityIdentitysequential_2/dense_10/Tanh:y:0*
T0*'
_output_shapes
:����������
+sequential_2/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential_2/dense_11/MatMulMatMul(sequential_2/dropout_2/Identity:output:03sequential_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,sequential_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_2/dense_11/BiasAddBiasAdd&sequential_2/dense_11/MatMul:product:04sequential_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
sequential_2/dense_11/TanhTanh&sequential_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:����������
+sequential_2/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_12_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
sequential_2/dense_12/MatMulMatMulsequential_2/dense_11/Tanh:y:03sequential_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
,sequential_2/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
sequential_2/dense_12/BiasAddBiasAdd&sequential_2/dense_12/MatMul:product:04sequential_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
|
sequential_2/dense_12/TanhTanh&sequential_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
+sequential_2/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_13_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
sequential_2/dense_13/MatMulMatMulsequential_2/dense_12/Tanh:y:03sequential_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,sequential_2/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_2/dense_13/BiasAddBiasAdd&sequential_2/dense_13/MatMul:product:04sequential_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
sequential_2/dense_13/TanhTanh&sequential_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:����������
+sequential_2/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_14_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential_2/dense_14/MatMulMatMulsequential_2/dense_13/Tanh:y:03sequential_2/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,sequential_2/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_2/dense_14/BiasAddBiasAdd&sequential_2/dense_14/MatMul:product:04sequential_2/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������u
IdentityIdentity&sequential_2/dense_14/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^sequential_2/dense_10/BiasAdd/ReadVariableOp,^sequential_2/dense_10/MatMul/ReadVariableOp-^sequential_2/dense_11/BiasAdd/ReadVariableOp,^sequential_2/dense_11/MatMul/ReadVariableOp-^sequential_2/dense_12/BiasAdd/ReadVariableOp,^sequential_2/dense_12/MatMul/ReadVariableOp-^sequential_2/dense_13/BiasAdd/ReadVariableOp,^sequential_2/dense_13/MatMul/ReadVariableOp-^sequential_2/dense_14/BiasAdd/ReadVariableOp,^sequential_2/dense_14/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������::: : : : : : : : : : 2\
,sequential_2/dense_10/BiasAdd/ReadVariableOp,sequential_2/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_10/MatMul/ReadVariableOp+sequential_2/dense_10/MatMul/ReadVariableOp2\
,sequential_2/dense_11/BiasAdd/ReadVariableOp,sequential_2/dense_11/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_11/MatMul/ReadVariableOp+sequential_2/dense_11/MatMul/ReadVariableOp2\
,sequential_2/dense_12/BiasAdd/ReadVariableOp,sequential_2/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_12/MatMul/ReadVariableOp+sequential_2/dense_12/MatMul/ReadVariableOp2\
,sequential_2/dense_13/BiasAdd/ReadVariableOp,sequential_2/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_13/MatMul/ReadVariableOp+sequential_2/dense_13/MatMul/ReadVariableOp2\
,sequential_2/dense_14/BiasAdd/ReadVariableOp,sequential_2/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_14/MatMul/ReadVariableOp+sequential_2/dense_14/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:$ 

_output_shapes

::$ 

_output_shapes

::^ Z
'
_output_shapes
:���������
/
_user_specified_namenormalization_2_input"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
W
normalization_2_input>
'serving_default_normalization_2_input:0���������<
dense_140
StatefulPartitionedCall:0���������tensorflow/serving/predict:إ
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias"
_tf_keras_layer
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias"
_tf_keras_layer
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias"
_tf_keras_layer
~
0
1
2
3
 4
.5
/6
67
78
>9
?10
F11
G12"
trackable_list_wrapper
f
0
 1
.2
/3
64
75
>6
?7
F8
G9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Mtrace_0
Ntrace_12�
.__inference_sequential_2_layer_call_fn_4874114
.__inference_sequential_2_layer_call_fn_4874143�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zMtrace_0zNtrace_1
�
Otrace_0
Ptrace_12�
I__inference_sequential_2_layer_call_and_return_conditional_losses_4874042
I__inference_sequential_2_layer_call_and_return_conditional_losses_4874085�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zOtrace_0zPtrace_1
�
Q	capture_0
R	capture_1B�
"__inference__wrapped_model_4873938normalization_2_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zQ	capture_0zR	capture_1
�
S
_variables
T_iterations
U_learning_rate
V_index_dict
W
_momentums
X_velocities
Y_update_step_xla"
experimentalOptimizer
,
Zserving_default"
signature_map
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 :2normalization_2/mean
$:"2normalization_2/variance
:	 2normalization_2/count
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
`trace_02�
*__inference_dense_10_layer_call_fn_4874239�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`trace_0
�
atrace_02�
E__inference_dense_10_layer_call_and_return_conditional_losses_4874250�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zatrace_0
!:2dense_10/kernel
:2dense_10/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
gtrace_0
htrace_12�
+__inference_dropout_2_layer_call_fn_4874255
+__inference_dropout_2_layer_call_fn_4874260�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zgtrace_0zhtrace_1
�
itrace_0
jtrace_12�
F__inference_dropout_2_layer_call_and_return_conditional_losses_4874272
F__inference_dropout_2_layer_call_and_return_conditional_losses_4874277�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zitrace_0zjtrace_1
"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
ptrace_02�
*__inference_dense_11_layer_call_fn_4874286�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zptrace_0
�
qtrace_02�
E__inference_dense_11_layer_call_and_return_conditional_losses_4874297�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zqtrace_0
!:2dense_11/kernel
:2dense_11/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
wtrace_02�
*__inference_dense_12_layer_call_fn_4874306�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zwtrace_0
�
xtrace_02�
E__inference_dense_12_layer_call_and_return_conditional_losses_4874317�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zxtrace_0
!:
2dense_12/kernel
:
2dense_12/bias
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
�
~trace_02�
*__inference_dense_13_layer_call_fn_4874326�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z~trace_0
�
trace_02�
E__inference_dense_13_layer_call_and_return_conditional_losses_4874337�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
!:
2dense_13/kernel
:2dense_13/bias
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_14_layer_call_fn_4874346�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_14_layer_call_and_return_conditional_losses_4874356�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:2dense_14/kernel
:2dense_14/bias
5
0
1
2"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
Q	capture_0
R	capture_1B�
.__inference_sequential_2_layer_call_fn_4874114normalization_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zQ	capture_0zR	capture_1
�
Q	capture_0
R	capture_1B�
.__inference_sequential_2_layer_call_fn_4874143normalization_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zQ	capture_0zR	capture_1
�
Q	capture_0
R	capture_1B�
I__inference_sequential_2_layer_call_and_return_conditional_losses_4874042normalization_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zQ	capture_0zR	capture_1
�
Q	capture_0
R	capture_1B�
I__inference_sequential_2_layer_call_and_return_conditional_losses_4874085normalization_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zQ	capture_0zR	capture_1
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
�
T0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
p
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9"
trackable_list_wrapper
p
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�
Q	capture_0
R	capture_1B�
%__inference_signature_wrapper_4874230normalization_2_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zQ	capture_0zR	capture_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_10_layer_call_fn_4874239inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_10_layer_call_and_return_conditional_losses_4874250inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dropout_2_layer_call_fn_4874255inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_2_layer_call_fn_4874260inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_2_layer_call_and_return_conditional_losses_4874272inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_2_layer_call_and_return_conditional_losses_4874277inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_11_layer_call_fn_4874286inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_11_layer_call_and_return_conditional_losses_4874297inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_12_layer_call_fn_4874306inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_12_layer_call_and_return_conditional_losses_4874317inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_13_layer_call_fn_4874326inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_13_layer_call_and_return_conditional_losses_4874337inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_14_layer_call_fn_4874346inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_14_layer_call_and_return_conditional_losses_4874356inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
&:$2Adam/m/dense_10/kernel
&:$2Adam/v/dense_10/kernel
 :2Adam/m/dense_10/bias
 :2Adam/v/dense_10/bias
&:$2Adam/m/dense_11/kernel
&:$2Adam/v/dense_11/kernel
 :2Adam/m/dense_11/bias
 :2Adam/v/dense_11/bias
&:$
2Adam/m/dense_12/kernel
&:$
2Adam/v/dense_12/kernel
 :
2Adam/m/dense_12/bias
 :
2Adam/v/dense_12/bias
&:$
2Adam/m/dense_13/kernel
&:$
2Adam/v/dense_13/kernel
 :2Adam/m/dense_13/bias
 :2Adam/v/dense_13/bias
&:$2Adam/m/dense_14/kernel
&:$2Adam/v/dense_14/kernel
 :2Adam/m/dense_14/bias
 :2Adam/v/dense_14/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count�
"__inference__wrapped_model_4873938�QR ./67>?FG>�;
4�1
/�,
normalization_2_input���������
� "3�0
.
dense_14"�
dense_14����������
E__inference_dense_10_layer_call_and_return_conditional_losses_4874250c /�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
*__inference_dense_10_layer_call_fn_4874239X /�,
%�"
 �
inputs���������
� "!�
unknown����������
E__inference_dense_11_layer_call_and_return_conditional_losses_4874297c.//�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
*__inference_dense_11_layer_call_fn_4874286X.//�,
%�"
 �
inputs���������
� "!�
unknown����������
E__inference_dense_12_layer_call_and_return_conditional_losses_4874317c67/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������

� �
*__inference_dense_12_layer_call_fn_4874306X67/�,
%�"
 �
inputs���������
� "!�
unknown���������
�
E__inference_dense_13_layer_call_and_return_conditional_losses_4874337c>?/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������
� �
*__inference_dense_13_layer_call_fn_4874326X>?/�,
%�"
 �
inputs���������

� "!�
unknown����������
E__inference_dense_14_layer_call_and_return_conditional_losses_4874356cFG/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
*__inference_dense_14_layer_call_fn_4874346XFG/�,
%�"
 �
inputs���������
� "!�
unknown����������
F__inference_dropout_2_layer_call_and_return_conditional_losses_4874272c3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
F__inference_dropout_2_layer_call_and_return_conditional_losses_4874277c3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
+__inference_dropout_2_layer_call_fn_4874255X3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
+__inference_dropout_2_layer_call_fn_4874260X3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
I__inference_sequential_2_layer_call_and_return_conditional_losses_4874042�QR ./67>?FGF�C
<�9
/�,
normalization_2_input���������
p

 
� ",�)
"�
tensor_0���������
� �
I__inference_sequential_2_layer_call_and_return_conditional_losses_4874085�QR ./67>?FGF�C
<�9
/�,
normalization_2_input���������
p 

 
� ",�)
"�
tensor_0���������
� �
.__inference_sequential_2_layer_call_fn_4874114yQR ./67>?FGF�C
<�9
/�,
normalization_2_input���������
p

 
� "!�
unknown����������
.__inference_sequential_2_layer_call_fn_4874143yQR ./67>?FGF�C
<�9
/�,
normalization_2_input���������
p 

 
� "!�
unknown����������
%__inference_signature_wrapper_4874230�QR ./67>?FGW�T
� 
M�J
H
normalization_2_input/�,
normalization_2_input���������"3�0
.
dense_14"�
dense_14���������