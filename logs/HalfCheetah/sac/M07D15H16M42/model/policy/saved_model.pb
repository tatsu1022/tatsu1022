Æ
¢ò
:
Add
x"T
y"T
z"T"
Ttype:
2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
0
Neg
x"T
y"T"
Ttype:
2
	
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
8
Pow
x"T
y"T
z"T"
Ttype:
2
	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718õ
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

actor/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameactor/dense1/kernel
|
'actor/dense1/kernel/Read/ReadVariableOpReadVariableOpactor/dense1/kernel*
_output_shapes
:	*
dtype0
{
actor/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameactor/dense1/bias
t
%actor/dense1/bias/Read/ReadVariableOpReadVariableOpactor/dense1/bias*
_output_shapes	
:*
dtype0

actor/dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameactor/dense2/kernel
}
'actor/dense2/kernel/Read/ReadVariableOpReadVariableOpactor/dense2/kernel* 
_output_shapes
:
*
dtype0
{
actor/dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameactor/dense2/bias
t
%actor/dense2/bias/Read/ReadVariableOpReadVariableOpactor/dense2/bias*
_output_shapes	
:*
dtype0

actor/means/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_nameactor/means/kernel
z
&actor/means/kernel/Read/ReadVariableOpReadVariableOpactor/means/kernel*
_output_shapes
:	*
dtype0
x
actor/means/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameactor/means/bias
q
$actor/means/bias/Read/ReadVariableOpReadVariableOpactor/means/bias*
_output_shapes
:*
dtype0

actor/log_stdss/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameactor/log_stdss/kernel

*actor/log_stdss/kernel/Read/ReadVariableOpReadVariableOpactor/log_stdss/kernel*
_output_shapes
:	*
dtype0

actor/log_stdss/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameactor/log_stdss/bias
y
(actor/log_stdss/bias/Read/ReadVariableOpReadVariableOpactor/log_stdss/bias*
_output_shapes
:*
dtype0

Adam/actor/dense1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_nameAdam/actor/dense1/kernel/m

.Adam/actor/dense1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/actor/dense1/kernel/m*
_output_shapes
:	*
dtype0

Adam/actor/dense1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/actor/dense1/bias/m

,Adam/actor/dense1/bias/m/Read/ReadVariableOpReadVariableOpAdam/actor/dense1/bias/m*
_output_shapes	
:*
dtype0

Adam/actor/means/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_nameAdam/actor/means/kernel/m

-Adam/actor/means/kernel/m/Read/ReadVariableOpReadVariableOpAdam/actor/means/kernel/m*
_output_shapes
:	*
dtype0

Adam/actor/means/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/actor/means/bias/m

+Adam/actor/means/bias/m/Read/ReadVariableOpReadVariableOpAdam/actor/means/bias/m*
_output_shapes
:*
dtype0

Adam/actor/log_stdss/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*.
shared_nameAdam/actor/log_stdss/kernel/m

1Adam/actor/log_stdss/kernel/m/Read/ReadVariableOpReadVariableOpAdam/actor/log_stdss/kernel/m*
_output_shapes
:	*
dtype0

Adam/actor/log_stdss/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/actor/log_stdss/bias/m

/Adam/actor/log_stdss/bias/m/Read/ReadVariableOpReadVariableOpAdam/actor/log_stdss/bias/m*
_output_shapes
:*
dtype0

Adam/actor/dense1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_nameAdam/actor/dense1/kernel/v

.Adam/actor/dense1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/actor/dense1/kernel/v*
_output_shapes
:	*
dtype0

Adam/actor/dense1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/actor/dense1/bias/v

,Adam/actor/dense1/bias/v/Read/ReadVariableOpReadVariableOpAdam/actor/dense1/bias/v*
_output_shapes	
:*
dtype0

Adam/actor/means/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_nameAdam/actor/means/kernel/v

-Adam/actor/means/kernel/v/Read/ReadVariableOpReadVariableOpAdam/actor/means/kernel/v*
_output_shapes
:	*
dtype0

Adam/actor/means/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/actor/means/bias/v

+Adam/actor/means/bias/v/Read/ReadVariableOpReadVariableOpAdam/actor/means/bias/v*
_output_shapes
:*
dtype0

Adam/actor/log_stdss/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*.
shared_nameAdam/actor/log_stdss/kernel/v

1Adam/actor/log_stdss/kernel/v/Read/ReadVariableOpReadVariableOpAdam/actor/log_stdss/kernel/v*
_output_shapes
:	*
dtype0

Adam/actor/log_stdss/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/actor/log_stdss/bias/v

/Adam/actor/log_stdss/bias/v/Read/ReadVariableOpReadVariableOpAdam/actor/log_stdss/bias/v*
_output_shapes
:*
dtype0
N
ConstConst*
_output_shapes
: *
dtype0*
valueB 2      ð?
P
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2        

NoOpNoOp
Á#
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*ú"
valueð"Bí" Bæ"
´
hidden_units
	optimizer

dense1

dense2
	means
log_stds
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
 
¬
iter

beta_1

beta_2
	decay
learning_ratemBmCmDmE#mF$mGvHvIvJvK#vL$vM
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
h

#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
8
0
1
2
3
4
5
#6
$7
 
8
0
1
2
3
4
5
#6
$7
­
	variables

)layers
*non_trainable_variables
+layer_metrics
,metrics
-layer_regularization_losses
regularization_losses
	trainable_variables
 
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEactor/dense1/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEactor/dense1/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
	variables
regularization_losses

.layers
/layer_metrics
0metrics
1layer_regularization_losses
2non_trainable_variables
trainable_variables
QO
VARIABLE_VALUEactor/dense2/kernel(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEactor/dense2/bias&dense2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
	variables
regularization_losses

3layers
4layer_metrics
5metrics
6layer_regularization_losses
7non_trainable_variables
trainable_variables
OM
VARIABLE_VALUEactor/means/kernel'means/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEactor/means/bias%means/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
	variables
 regularization_losses

8layers
9layer_metrics
:metrics
;layer_regularization_losses
<non_trainable_variables
!trainable_variables
VT
VARIABLE_VALUEactor/log_stdss/kernel*log_stds/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEactor/log_stdss/bias(log_stds/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1
­
%	variables
&regularization_losses

=layers
>layer_metrics
?metrics
@layer_regularization_losses
Anon_trainable_variables
'trainable_variables

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
tr
VARIABLE_VALUEAdam/actor/dense1/kernel/mDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/actor/dense1/bias/mBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/actor/means/kernel/mCmeans/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/actor/means/bias/mAmeans/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/actor/log_stdss/kernel/mFlog_stds/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/actor/log_stdss/bias/mDlog_stds/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/actor/dense1/kernel/vDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/actor/dense1/bias/vBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/actor/means/kernel/vCmeans/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/actor/means/bias/vAmeans/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/actor/log_stdss/kernel/vFlog_stds/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/actor/log_stdss/bias/vDlog_stds/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
§
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1actor/dense1/kernelactor/dense1/biasactor/dense2/kernelactor/dense2/biasactor/means/kernelactor/means/biasactor/log_stdss/kernelactor/log_stdss/biasConstConst_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 */
f*R(
&__inference_signature_wrapper_33364792
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
æ

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'actor/dense1/kernel/Read/ReadVariableOp%actor/dense1/bias/Read/ReadVariableOp'actor/dense2/kernel/Read/ReadVariableOp%actor/dense2/bias/Read/ReadVariableOp&actor/means/kernel/Read/ReadVariableOp$actor/means/bias/Read/ReadVariableOp*actor/log_stdss/kernel/Read/ReadVariableOp(actor/log_stdss/bias/Read/ReadVariableOp.Adam/actor/dense1/kernel/m/Read/ReadVariableOp,Adam/actor/dense1/bias/m/Read/ReadVariableOp-Adam/actor/means/kernel/m/Read/ReadVariableOp+Adam/actor/means/bias/m/Read/ReadVariableOp1Adam/actor/log_stdss/kernel/m/Read/ReadVariableOp/Adam/actor/log_stdss/bias/m/Read/ReadVariableOp.Adam/actor/dense1/kernel/v/Read/ReadVariableOp,Adam/actor/dense1/bias/v/Read/ReadVariableOp-Adam/actor/means/kernel/v/Read/ReadVariableOp+Adam/actor/means/bias/v/Read/ReadVariableOp1Adam/actor/log_stdss/kernel/v/Read/ReadVariableOp/Adam/actor/log_stdss/bias/v/Read/ReadVariableOpConst_2*&
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__traced_save_33364972
ë
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateactor/dense1/kernelactor/dense1/biasactor/dense2/kernelactor/dense2/biasactor/means/kernelactor/means/biasactor/log_stdss/kernelactor/log_stdss/biasAdam/actor/dense1/kernel/mAdam/actor/dense1/bias/mAdam/actor/means/kernel/mAdam/actor/means/bias/mAdam/actor/log_stdss/kernel/mAdam/actor/log_stdss/bias/mAdam/actor/dense1/kernel/vAdam/actor/dense1/bias/vAdam/actor/means/kernel/vAdam/actor/means/bias/vAdam/actor/log_stdss/kernel/vAdam/actor/log_stdss/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference__traced_restore_33365057Õ
¡

(__inference_means_layer_call_fn_33364841

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_means_layer_call_and_return_conditional_losses_333646062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
¬
#__inference__wrapped_model_33364558
input_1>
+actor_dense1_matmul_readvariableop_resource:	;
,actor_dense1_biasadd_readvariableop_resource:	?
+actor_dense2_matmul_readvariableop_resource:
;
,actor_dense2_biasadd_readvariableop_resource:	=
*actor_means_matmul_readvariableop_resource:	9
+actor_means_biasadd_readvariableop_resource:A
.actor_log_stdss_matmul_readvariableop_resource:	=
/actor_log_stdss_biasadd_readvariableop_resource:
actor_mul_x
actor_add_y
identity

identity_1

identity_2¢#actor/dense1/BiasAdd/ReadVariableOp¢"actor/dense1/MatMul/ReadVariableOp¢#actor/dense2/BiasAdd/ReadVariableOp¢"actor/dense2/MatMul/ReadVariableOp¢&actor/log_stdss/BiasAdd/ReadVariableOp¢%actor/log_stdss/MatMul/ReadVariableOp¢"actor/means/BiasAdd/ReadVariableOp¢!actor/means/MatMul/ReadVariableOpµ
"actor/dense1/MatMul/ReadVariableOpReadVariableOp+actor_dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02$
"actor/dense1/MatMul/ReadVariableOp
actor/dense1/MatMulMatMulinput_1*actor/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense1/MatMul´
#actor/dense1/BiasAdd/ReadVariableOpReadVariableOp,actor_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#actor/dense1/BiasAdd/ReadVariableOp¶
actor/dense1/BiasAddBiasAddactor/dense1/MatMul:product:0+actor/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense1/BiasAdd
actor/dense1/ReluReluactor/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense1/Relu¶
"actor/dense2/MatMul/ReadVariableOpReadVariableOp+actor_dense2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02$
"actor/dense2/MatMul/ReadVariableOp´
actor/dense2/MatMulMatMulactor/dense1/Relu:activations:0*actor/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense2/MatMul´
#actor/dense2/BiasAdd/ReadVariableOpReadVariableOp,actor_dense2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#actor/dense2/BiasAdd/ReadVariableOp¶
actor/dense2/BiasAddBiasAddactor/dense2/MatMul:product:0+actor/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense2/BiasAdd
actor/dense2/ReluReluactor/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/dense2/Relu²
!actor/means/MatMul/ReadVariableOpReadVariableOp*actor_means_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!actor/means/MatMul/ReadVariableOp°
actor/means/MatMulMatMulactor/dense2/Relu:activations:0)actor/means/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/means/MatMul°
"actor/means/BiasAdd/ReadVariableOpReadVariableOp+actor_means_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"actor/means/BiasAdd/ReadVariableOp±
actor/means/BiasAddBiasAddactor/means/MatMul:product:0*actor/means/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/means/BiasAdd¾
%actor/log_stdss/MatMul/ReadVariableOpReadVariableOp.actor_log_stdss_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02'
%actor/log_stdss/MatMul/ReadVariableOp¼
actor/log_stdss/MatMulMatMulactor/dense2/Relu:activations:0-actor/log_stdss/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/log_stdss/MatMul¼
&actor/log_stdss/BiasAdd/ReadVariableOpReadVariableOp/actor_log_stdss_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&actor/log_stdss/BiasAdd/ReadVariableOpÁ
actor/log_stdss/BiasAddBiasAdd actor/log_stdss/MatMul:product:0.actor/log_stdss/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/log_stdss/BiasAddq
	actor/ExpExp actor/log_stdss/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	actor/Exp
&actor/actor_Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2(
&actor/actor_Normal/sample/sample_shape
actor/actor_Normal/sample/ShapeShapeactor/means/BiasAdd:output:0*
T0*
_output_shapes
:2!
actor/actor_Normal/sample/Shape¨
-actor/actor_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-actor/actor_Normal/sample/strided_slice/stack¬
/actor/actor_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/actor/actor_Normal/sample/strided_slice/stack_1¬
/actor/actor_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/actor/actor_Normal/sample/strided_slice/stack_2ü
'actor/actor_Normal/sample/strided_sliceStridedSlice(actor/actor_Normal/sample/Shape:output:06actor/actor_Normal/sample/strided_slice/stack:output:08actor/actor_Normal/sample/strided_slice/stack_1:output:08actor/actor_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'actor/actor_Normal/sample/strided_slice
!actor/actor_Normal/sample/Shape_1Shapeactor/Exp:y:0*
T0*
_output_shapes
:2#
!actor/actor_Normal/sample/Shape_1¬
/actor/actor_Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/actor/actor_Normal/sample/strided_slice_1/stack°
1actor/actor_Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1actor/actor_Normal/sample/strided_slice_1/stack_1°
1actor/actor_Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1actor/actor_Normal/sample/strided_slice_1/stack_2
)actor/actor_Normal/sample/strided_slice_1StridedSlice*actor/actor_Normal/sample/Shape_1:output:08actor/actor_Normal/sample/strided_slice_1/stack:output:0:actor/actor_Normal/sample/strided_slice_1/stack_1:output:0:actor/actor_Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2+
)actor/actor_Normal/sample/strided_slice_1
*actor/actor_Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2,
*actor/actor_Normal/sample/BroadcastArgs/s0
,actor/actor_Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2.
,actor/actor_Normal/sample/BroadcastArgs/s0_1è
'actor/actor_Normal/sample/BroadcastArgsBroadcastArgs5actor/actor_Normal/sample/BroadcastArgs/s0_1:output:00actor/actor_Normal/sample/strided_slice:output:0*
_output_shapes
:2)
'actor/actor_Normal/sample/BroadcastArgså
)actor/actor_Normal/sample/BroadcastArgs_1BroadcastArgs,actor/actor_Normal/sample/BroadcastArgs:r0:02actor/actor_Normal/sample/strided_slice_1:output:0*
_output_shapes
:2+
)actor/actor_Normal/sample/BroadcastArgs_1 
)actor/actor_Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2+
)actor/actor_Normal/sample/concat/values_0
%actor/actor_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%actor/actor_Normal/sample/concat/axis
 actor/actor_Normal/sample/concatConcatV22actor/actor_Normal/sample/concat/values_0:output:0.actor/actor_Normal/sample/BroadcastArgs_1:r0:0.actor/actor_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 actor/actor_Normal/sample/concat³
3actor/actor_Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB 2        25
3actor/actor_Normal/sample/normal/random_normal/mean·
5actor/actor_Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB 2      ð?27
5actor/actor_Normal/sample/normal/random_normal/stddev
Cactor/actor_Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal)actor/actor_Normal/sample/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02E
Cactor/actor_Normal/sample/normal/random_normal/RandomStandardNormal¼
2actor/actor_Normal/sample/normal/random_normal/mulMulLactor/actor_Normal/sample/normal/random_normal/RandomStandardNormal:output:0>actor/actor_Normal/sample/normal/random_normal/stddev:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ24
2actor/actor_Normal/sample/normal/random_normal/mul
.actor/actor_Normal/sample/normal/random_normalAdd6actor/actor_Normal/sample/normal/random_normal/mul:z:0<actor/actor_Normal/sample/normal/random_normal/mean:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ20
.actor/actor_Normal/sample/normal/random_normal¾
actor/actor_Normal/sample/mulMul2actor/actor_Normal/sample/normal/random_normal:z:0actor/Exp:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/actor_Normal/sample/mul¾
actor/actor_Normal/sample/addAddV2!actor/actor_Normal/sample/mul:z:0actor/means/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/actor_Normal/sample/add
!actor/actor_Normal/sample/Shape_2Shape!actor/actor_Normal/sample/add:z:0*
T0*
_output_shapes
:2#
!actor/actor_Normal/sample/Shape_2¬
/actor/actor_Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/actor/actor_Normal/sample/strided_slice_2/stack°
1actor/actor_Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1actor/actor_Normal/sample/strided_slice_2/stack_1°
1actor/actor_Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1actor/actor_Normal/sample/strided_slice_2/stack_2
)actor/actor_Normal/sample/strided_slice_2StridedSlice*actor/actor_Normal/sample/Shape_2:output:08actor/actor_Normal/sample/strided_slice_2/stack:output:0:actor/actor_Normal/sample/strided_slice_2/stack_1:output:0:actor/actor_Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2+
)actor/actor_Normal/sample/strided_slice_2
'actor/actor_Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'actor/actor_Normal/sample/concat_1/axis
"actor/actor_Normal/sample/concat_1ConcatV2/actor/actor_Normal/sample/sample_shape:output:02actor/actor_Normal/sample/strided_slice_2:output:00actor/actor_Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"actor/actor_Normal/sample/concat_1Ó
!actor/actor_Normal/sample/ReshapeReshape!actor/actor_Normal/sample/add:z:0+actor/actor_Normal/sample/concat_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!actor/actor_Normal/sample/Reshape~

actor/TanhTanh*actor/actor_Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

actor/Tanhl
	actor/mulMulactor_mul_xactor/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	actor/mulm
	actor/addAddV2actor/mul:z:0actor_add_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	actor/addÆ
%actor/actor_Normal_1/log_prob/truedivRealDiv*actor/actor_Normal/sample/Reshape:output:0actor/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%actor/actor_Normal_1/log_prob/truediv¼
'actor/actor_Normal_1/log_prob/truediv_1RealDivactor/means/BiasAdd:output:0actor/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'actor/actor_Normal_1/log_prob/truediv_1
/actor/actor_Normal_1/log_prob/SquaredDifferenceSquaredDifference)actor/actor_Normal_1/log_prob/truediv:z:0+actor/actor_Normal_1/log_prob/truediv_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/actor/actor_Normal_1/log_prob/SquaredDifference
#actor/actor_Normal_1/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB 2      à¿2%
#actor/actor_Normal_1/log_prob/mul/xâ
!actor/actor_Normal_1/log_prob/mulMul,actor/actor_Normal_1/log_prob/mul/x:output:03actor/actor_Normal_1/log_prob/SquaredDifference:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!actor/actor_Normal_1/log_prob/mul
#actor/actor_Normal_1/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB 2´¾dÈñgí?2%
#actor/actor_Normal_1/log_prob/Const
!actor/actor_Normal_1/log_prob/LogLogactor/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!actor/actor_Normal_1/log_prob/LogÖ
!actor/actor_Normal_1/log_prob/addAddV2,actor/actor_Normal_1/log_prob/Const:output:0%actor/actor_Normal_1/log_prob/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!actor/actor_Normal_1/log_prob/addÍ
!actor/actor_Normal_1/log_prob/subSub%actor/actor_Normal_1/log_prob/mul:z:0%actor/actor_Normal_1/log_prob/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!actor/actor_Normal_1/log_prob/subc
actor/pow/yConst*
_output_shapes
: *
dtype0*
valueB 2       @2
actor/pow/yt
	actor/powPowactor/add:z:0actor/pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	actor/powc
actor/sub/xConst*
_output_shapes
: *
dtype0*
valueB 2      ð?2
actor/sub/xt
	actor/subSubactor/sub/x:output:0actor/pow:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	actor/subg
actor/add_1/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
actor/add_1/y|
actor/add_1AddV2actor/sub:z:0actor/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/add_1`
	actor/LogLogactor/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	actor/Log
actor/sub_1Sub%actor/actor_Normal_1/log_prob/sub:z:0actor/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/sub_1|
actor/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
actor/Sum/reduction_indices
	actor/SumSumactor/sub_1:z:0$actor/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
	actor/Sumc
	actor/NegNegactor/Sum:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	actor/Negt
actor/Tanh_1Tanhactor/means/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor/Tanh_1
IdentityIdentityactor/add:z:0$^actor/dense1/BiasAdd/ReadVariableOp#^actor/dense1/MatMul/ReadVariableOp$^actor/dense2/BiasAdd/ReadVariableOp#^actor/dense2/MatMul/ReadVariableOp'^actor/log_stdss/BiasAdd/ReadVariableOp&^actor/log_stdss/MatMul/ReadVariableOp#^actor/means/BiasAdd/ReadVariableOp"^actor/means/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identityactor/Neg:y:0$^actor/dense1/BiasAdd/ReadVariableOp#^actor/dense1/MatMul/ReadVariableOp$^actor/dense2/BiasAdd/ReadVariableOp#^actor/dense2/MatMul/ReadVariableOp'^actor/log_stdss/BiasAdd/ReadVariableOp&^actor/log_stdss/MatMul/ReadVariableOp#^actor/means/BiasAdd/ReadVariableOp"^actor/means/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identityactor/Tanh_1:y:0$^actor/dense1/BiasAdd/ReadVariableOp#^actor/dense1/MatMul/ReadVariableOp$^actor/dense2/BiasAdd/ReadVariableOp#^actor/dense2/MatMul/ReadVariableOp'^actor/log_stdss/BiasAdd/ReadVariableOp&^actor/log_stdss/MatMul/ReadVariableOp#^actor/means/BiasAdd/ReadVariableOp"^actor/means/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2J
#actor/dense1/BiasAdd/ReadVariableOp#actor/dense1/BiasAdd/ReadVariableOp2H
"actor/dense1/MatMul/ReadVariableOp"actor/dense1/MatMul/ReadVariableOp2J
#actor/dense2/BiasAdd/ReadVariableOp#actor/dense2/BiasAdd/ReadVariableOp2H
"actor/dense2/MatMul/ReadVariableOp"actor/dense2/MatMul/ReadVariableOp2P
&actor/log_stdss/BiasAdd/ReadVariableOp&actor/log_stdss/BiasAdd/ReadVariableOp2N
%actor/log_stdss/MatMul/ReadVariableOp%actor/log_stdss/MatMul/ReadVariableOp2H
"actor/means/BiasAdd/ReadVariableOp"actor/means/BiasAdd/ReadVariableOp2F
!actor/means/MatMul/ReadVariableOp!actor/means/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:	

_output_shapes
: :


_output_shapes
: 
ëb
²
C__inference_actor_layer_call_and_return_conditional_losses_33364691
input_1"
dense1_33364574:	
dense1_33364576:	#
dense2_33364591:

dense2_33364593:	!
means_33364607:	
means_33364609:%
log_stdss_33364623:	 
log_stdss_33364625:	
mul_x	
add_y
identity

identity_1

identity_2¢dense1/StatefulPartitionedCall¢dense2/StatefulPartitionedCall¢!log_stdss/StatefulPartitionedCall¢means/StatefulPartitionedCall
dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_33364574dense1_33364576*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_333645732 
dense1/StatefulPartitionedCall·
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_33364591dense2_33364593*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense2_layer_call_and_return_conditional_losses_333645902 
dense2/StatefulPartitionedCall±
means/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0means_33364607means_33364609*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_means_layer_call_and_return_conditional_losses_333646062
means/StatefulPartitionedCallÅ
!log_stdss/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0log_stdss_33364623log_stdss_33364625*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_log_stdss_layer_call_and_return_conditional_losses_333646222#
!log_stdss/StatefulPartitionedCallo
ExpExp*log_stdss/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Exp
Normal_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2
Normal_1/sample/sample_shape
Normal_1/sample/ShapeShape&means/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
Normal_1/sample/Shape
#Normal_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#Normal_1/sample/strided_slice/stack
%Normal_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Normal_1/sample/strided_slice/stack_1
%Normal_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Normal_1/sample/strided_slice/stack_2À
Normal_1/sample/strided_sliceStridedSliceNormal_1/sample/Shape:output:0,Normal_1/sample/strided_slice/stack:output:0.Normal_1/sample/strided_slice/stack_1:output:0.Normal_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
Normal_1/sample/strided_slicei
Normal_1/sample/Shape_1ShapeExp:y:0*
T0*
_output_shapes
:2
Normal_1/sample/Shape_1
%Normal_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Normal_1/sample/strided_slice_1/stack
'Normal_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Normal_1/sample/strided_slice_1/stack_1
'Normal_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Normal_1/sample/strided_slice_1/stack_2Ì
Normal_1/sample/strided_slice_1StridedSlice Normal_1/sample/Shape_1:output:0.Normal_1/sample/strided_slice_1/stack:output:00Normal_1/sample/strided_slice_1/stack_1:output:00Normal_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
Normal_1/sample/strided_slice_1
 Normal_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2"
 Normal_1/sample/BroadcastArgs/s0
"Normal_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2$
"Normal_1/sample/BroadcastArgs/s0_1À
Normal_1/sample/BroadcastArgsBroadcastArgs+Normal_1/sample/BroadcastArgs/s0_1:output:0&Normal_1/sample/strided_slice:output:0*
_output_shapes
:2
Normal_1/sample/BroadcastArgs½
Normal_1/sample/BroadcastArgs_1BroadcastArgs"Normal_1/sample/BroadcastArgs:r0:0(Normal_1/sample/strided_slice_1:output:0*
_output_shapes
:2!
Normal_1/sample/BroadcastArgs_1
Normal_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2!
Normal_1/sample/concat/values_0|
Normal_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Normal_1/sample/concat/axisà
Normal_1/sample/concatConcatV2(Normal_1/sample/concat/values_0:output:0$Normal_1/sample/BroadcastArgs_1:r0:0$Normal_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Normal_1/sample/concat
)Normal_1/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB 2        2+
)Normal_1/sample/normal/random_normal/mean£
+Normal_1/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB 2      ð?2-
+Normal_1/sample/normal/random_normal/stddevû
9Normal_1/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal_1/sample/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02;
9Normal_1/sample/normal/random_normal/RandomStandardNormal
(Normal_1/sample/normal/random_normal/mulMulBNormal_1/sample/normal/random_normal/RandomStandardNormal:output:04Normal_1/sample/normal/random_normal/stddev:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(Normal_1/sample/normal/random_normal/mulô
$Normal_1/sample/normal/random_normalAdd,Normal_1/sample/normal/random_normal/mul:z:02Normal_1/sample/normal/random_normal/mean:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2&
$Normal_1/sample/normal/random_normal
Normal_1/sample/mulMul(Normal_1/sample/normal/random_normal:z:0Exp:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Normal_1/sample/mulª
Normal_1/sample/addAddV2Normal_1/sample/mul:z:0&means/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Normal_1/sample/addy
Normal_1/sample/Shape_2ShapeNormal_1/sample/add:z:0*
T0*
_output_shapes
:2
Normal_1/sample/Shape_2
%Normal_1/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%Normal_1/sample/strided_slice_2/stack
'Normal_1/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'Normal_1/sample/strided_slice_2/stack_1
'Normal_1/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Normal_1/sample/strided_slice_2/stack_2Ê
Normal_1/sample/strided_slice_2StridedSlice Normal_1/sample/Shape_2:output:0.Normal_1/sample/strided_slice_2/stack:output:00Normal_1/sample/strided_slice_2/stack_1:output:00Normal_1/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2!
Normal_1/sample/strided_slice_2
Normal_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Normal_1/sample/concat_1/axisç
Normal_1/sample/concat_1ConcatV2%Normal_1/sample/sample_shape:output:0(Normal_1/sample/strided_slice_2:output:0&Normal_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Normal_1/sample/concat_1«
Normal_1/sample/ReshapeReshapeNormal_1/sample/add:z:0!Normal_1/sample/concat_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Normal_1/sample/Reshapeh
TanhTanh Normal_1/sample/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
TanhT
mulMulmul_xTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulU
addAddV2mul:z:0add_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
Normal_2/log_prob/truedivRealDiv Normal_1/sample/Reshape:output:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Normal_2/log_prob/truediv¨
Normal_2/log_prob/truediv_1RealDiv&means/StatefulPartitionedCall:output:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Normal_2/log_prob/truediv_1Ñ
#Normal_2/log_prob/SquaredDifferenceSquaredDifferenceNormal_2/log_prob/truediv:z:0Normal_2/log_prob/truediv_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#Normal_2/log_prob/SquaredDifference{
Normal_2/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB 2      à¿2
Normal_2/log_prob/mul/x²
Normal_2/log_prob/mulMul Normal_2/log_prob/mul/x:output:0'Normal_2/log_prob/SquaredDifference:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Normal_2/log_prob/mul{
Normal_2/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB 2´¾dÈñgí?2
Normal_2/log_prob/Constp
Normal_2/log_prob/LogLogExp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Normal_2/log_prob/Log¦
Normal_2/log_prob/addAddV2 Normal_2/log_prob/Const:output:0Normal_2/log_prob/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Normal_2/log_prob/add
Normal_2/log_prob/subSubNormal_2/log_prob/mul:z:0Normal_2/log_prob/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Normal_2/log_prob/subW
pow/yConst*
_output_shapes
: *
dtype0*
valueB 2       @2
pow/y\
powPowadd:z:0pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
powW
sub/xConst*
_output_shapes
: *
dtype0*
valueB 2      ð?2
sub/x\
subSubsub/x:output:0pow:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub[
add_1/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2	
add_1/yd
add_1AddV2sub:z:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1N
LogLog	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Logk
sub_1SubNormal_2/log_prob/sub:z:0Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_1p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices
SumSum	sub_1:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
SumQ
NegNegSum:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Negr
Tanh_1Tanh&means/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_1á
IdentityIdentityadd:z:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall"^log_stdss/StatefulPartitionedCall^means/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityå

Identity_1IdentityNeg:y:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall"^log_stdss/StatefulPartitionedCall^means/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1è

Identity_2Identity
Tanh_1:y:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall"^log_stdss/StatefulPartitionedCall^means/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2F
!log_stdss/StatefulPartitionedCall!log_stdss/StatefulPartitionedCall2>
means/StatefulPartitionedCallmeans/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:	

_output_shapes
: :


_output_shapes
: 
Þ:
¬
!__inference__traced_save_33364972
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_actor_dense1_kernel_read_readvariableop0
,savev2_actor_dense1_bias_read_readvariableop2
.savev2_actor_dense2_kernel_read_readvariableop0
,savev2_actor_dense2_bias_read_readvariableop1
-savev2_actor_means_kernel_read_readvariableop/
+savev2_actor_means_bias_read_readvariableop5
1savev2_actor_log_stdss_kernel_read_readvariableop3
/savev2_actor_log_stdss_bias_read_readvariableop9
5savev2_adam_actor_dense1_kernel_m_read_readvariableop7
3savev2_adam_actor_dense1_bias_m_read_readvariableop8
4savev2_adam_actor_means_kernel_m_read_readvariableop6
2savev2_adam_actor_means_bias_m_read_readvariableop<
8savev2_adam_actor_log_stdss_kernel_m_read_readvariableop:
6savev2_adam_actor_log_stdss_bias_m_read_readvariableop9
5savev2_adam_actor_dense1_kernel_v_read_readvariableop7
3savev2_adam_actor_dense1_bias_v_read_readvariableop8
4savev2_adam_actor_means_kernel_v_read_readvariableop6
2savev2_adam_actor_means_bias_v_read_readvariableop<
8savev2_adam_actor_log_stdss_kernel_v_read_readvariableop:
6savev2_adam_actor_log_stdss_bias_v_read_readvariableop
savev2_const_2

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¢
valueBB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB'means/kernel/.ATTRIBUTES/VARIABLE_VALUEB%means/bias/.ATTRIBUTES/VARIABLE_VALUEB*log_stds/kernel/.ATTRIBUTES/VARIABLE_VALUEB(log_stds/bias/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCmeans/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAmeans/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFlog_stds/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlog_stds/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCmeans/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAmeans/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFlog_stds/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlog_stds/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¼
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices­
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_actor_dense1_kernel_read_readvariableop,savev2_actor_dense1_bias_read_readvariableop.savev2_actor_dense2_kernel_read_readvariableop,savev2_actor_dense2_bias_read_readvariableop-savev2_actor_means_kernel_read_readvariableop+savev2_actor_means_bias_read_readvariableop1savev2_actor_log_stdss_kernel_read_readvariableop/savev2_actor_log_stdss_bias_read_readvariableop5savev2_adam_actor_dense1_kernel_m_read_readvariableop3savev2_adam_actor_dense1_bias_m_read_readvariableop4savev2_adam_actor_means_kernel_m_read_readvariableop2savev2_adam_actor_means_bias_m_read_readvariableop8savev2_adam_actor_log_stdss_kernel_m_read_readvariableop6savev2_adam_actor_log_stdss_bias_m_read_readvariableop5savev2_adam_actor_dense1_kernel_v_read_readvariableop3savev2_adam_actor_dense1_bias_v_read_readvariableop4savev2_adam_actor_means_kernel_v_read_readvariableop2savev2_adam_actor_means_bias_v_read_readvariableop8savev2_adam_actor_log_stdss_kernel_v_read_readvariableop6savev2_adam_actor_log_stdss_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ò
_input_shapesÀ
½: : : : : : :	::
::	::	::	::	::	::	::	::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!	

_output_shapes	
::%
!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: 
°
û
&__inference_signature_wrapper_33364792
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
	unknown_7
	unknown_8
identity

identity_1

identity_2¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *,
f'R%
#__inference__wrapped_model_333645582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:	

_output_shapes
: :


_output_shapes
: 
×	
ù
G__inference_log_stdss_layer_call_and_return_conditional_losses_33364622

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©

,__inference_log_stdss_layer_call_fn_33364860

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_log_stdss_layer_call_and_return_conditional_losses_333646222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ßl

$__inference__traced_restore_33365057
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: 9
&assignvariableop_5_actor_dense1_kernel:	3
$assignvariableop_6_actor_dense1_bias:	:
&assignvariableop_7_actor_dense2_kernel:
3
$assignvariableop_8_actor_dense2_bias:	8
%assignvariableop_9_actor_means_kernel:	2
$assignvariableop_10_actor_means_bias:=
*assignvariableop_11_actor_log_stdss_kernel:	6
(assignvariableop_12_actor_log_stdss_bias:A
.assignvariableop_13_adam_actor_dense1_kernel_m:	;
,assignvariableop_14_adam_actor_dense1_bias_m:	@
-assignvariableop_15_adam_actor_means_kernel_m:	9
+assignvariableop_16_adam_actor_means_bias_m:D
1assignvariableop_17_adam_actor_log_stdss_kernel_m:	=
/assignvariableop_18_adam_actor_log_stdss_bias_m:A
.assignvariableop_19_adam_actor_dense1_kernel_v:	;
,assignvariableop_20_adam_actor_dense1_bias_v:	@
-assignvariableop_21_adam_actor_means_kernel_v:	9
+assignvariableop_22_adam_actor_means_bias_v:D
1assignvariableop_23_adam_actor_log_stdss_kernel_v:	=
/assignvariableop_24_adam_actor_log_stdss_bias_v:
identity_26¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¢
valueBB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB'means/kernel/.ATTRIBUTES/VARIABLE_VALUEB%means/bias/.ATTRIBUTES/VARIABLE_VALUEB*log_stds/kernel/.ATTRIBUTES/VARIABLE_VALUEB(log_stds/bias/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCmeans/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAmeans/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFlog_stds/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlog_stds/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCmeans/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAmeans/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFlog_stds/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlog_stds/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÂ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices­
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2£
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¢
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ª
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5«
AssignVariableOp_5AssignVariableOp&assignvariableop_5_actor_dense1_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6©
AssignVariableOp_6AssignVariableOp$assignvariableop_6_actor_dense1_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7«
AssignVariableOp_7AssignVariableOp&assignvariableop_7_actor_dense2_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8©
AssignVariableOp_8AssignVariableOp$assignvariableop_8_actor_dense2_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ª
AssignVariableOp_9AssignVariableOp%assignvariableop_9_actor_means_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¬
AssignVariableOp_10AssignVariableOp$assignvariableop_10_actor_means_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11²
AssignVariableOp_11AssignVariableOp*assignvariableop_11_actor_log_stdss_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12°
AssignVariableOp_12AssignVariableOp(assignvariableop_12_actor_log_stdss_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¶
AssignVariableOp_13AssignVariableOp.assignvariableop_13_adam_actor_dense1_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14´
AssignVariableOp_14AssignVariableOp,assignvariableop_14_adam_actor_dense1_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15µ
AssignVariableOp_15AssignVariableOp-assignvariableop_15_adam_actor_means_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16³
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_actor_means_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¹
AssignVariableOp_17AssignVariableOp1assignvariableop_17_adam_actor_log_stdss_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18·
AssignVariableOp_18AssignVariableOp/assignvariableop_18_adam_actor_log_stdss_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¶
AssignVariableOp_19AssignVariableOp.assignvariableop_19_adam_actor_dense1_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20´
AssignVariableOp_20AssignVariableOp,assignvariableop_20_adam_actor_dense1_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21µ
AssignVariableOp_21AssignVariableOp-assignvariableop_21_adam_actor_means_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22³
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_actor_means_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¹
AssignVariableOp_23AssignVariableOp1assignvariableop_23_adam_actor_log_stdss_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24·
AssignVariableOp_24AssignVariableOp/assignvariableop_24_adam_actor_log_stdss_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25÷
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_26"#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¸

ø
D__inference_dense2_layer_call_and_return_conditional_losses_33364590

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

)__inference_dense2_layer_call_fn_33364821

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense2_layer_call_and_return_conditional_losses_333645902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´

÷
D__inference_dense1_layer_call_and_return_conditional_losses_33364573

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó	
õ
C__inference_means_layer_call_and_return_conditional_losses_33364606

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´

÷
D__inference_dense1_layer_call_and_return_conditional_losses_33364812

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×	
ù
G__inference_log_stdss_layer_call_and_return_conditional_losses_33364870

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸

ø
D__inference_dense2_layer_call_and_return_conditional_losses_33364832

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
ý
(__inference_actor_layer_call_fn_33364721
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:
	unknown_7
	unknown_8
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_333646912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:	

_output_shapes
: :


_output_shapes
: 
Ó	
õ
C__inference_means_layer_call_and_return_conditional_losses_33364851

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

)__inference_dense1_layer_call_fn_33364801

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_333645732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*§
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ<
output_20
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿ<
output_30
StatefulPartitionedCall:2ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ìq

hidden_units
	optimizer

dense1

dense2
	means
log_stds
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
N__call__
O_default_save_signature
*P&call_and_return_all_conditional_losses"
_tf_keras_modelõ{"name": "actor", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Actor", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 17]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Actor"}}
 "
trackable_list_wrapper
¿
iter

beta_1

beta_2
	decay
learning_ratemBmCmDmE#mF$mGvHvIvJvK#vL$vM"
	optimizer
È

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"£
_tf_keras_layer{"name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 17}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 17]}}
Ê

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
S__call__
*T&call_and_return_all_conditional_losses"¥
_tf_keras_layer{"name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
Ê

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
U__call__
*V&call_and_return_all_conditional_losses"¥
_tf_keras_layer{"name": "means", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "means", "trainable": true, "dtype": "float64", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
Ô

#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
W__call__
*X&call_and_return_all_conditional_losses"¯
_tf_keras_layer{"name": "log_stdss", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "log_stdss", "trainable": true, "dtype": "float64", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 15}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
X
0
1
2
3
4
5
#6
$7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
#6
$7"
trackable_list_wrapper
Ê
	variables

)layers
*non_trainable_variables
+layer_metrics
,metrics
-layer_regularization_losses
regularization_losses
	trainable_variables
N__call__
O_default_save_signature
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
,
Yserving_default"
signature_map
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
&:$	2actor/dense1/kernel
 :2actor/dense1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
	variables
regularization_losses

.layers
/layer_metrics
0metrics
1layer_regularization_losses
2non_trainable_variables
trainable_variables
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
':%
2actor/dense2/kernel
 :2actor/dense2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
	variables
regularization_losses

3layers
4layer_metrics
5metrics
6layer_regularization_losses
7non_trainable_variables
trainable_variables
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
%:#	2actor/means/kernel
:2actor/means/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
	variables
 regularization_losses

8layers
9layer_metrics
:metrics
;layer_regularization_losses
<non_trainable_variables
!trainable_variables
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
):'	2actor/log_stdss/kernel
": 2actor/log_stdss/bias
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
­
%	variables
&regularization_losses

=layers
>layer_metrics
?metrics
@layer_regularization_losses
Anon_trainable_variables
'trainable_variables
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
+:)	2Adam/actor/dense1/kernel/m
%:#2Adam/actor/dense1/bias/m
*:(	2Adam/actor/means/kernel/m
#:!2Adam/actor/means/bias/m
.:,	2Adam/actor/log_stdss/kernel/m
':%2Adam/actor/log_stdss/bias/m
+:)	2Adam/actor/dense1/kernel/v
%:#2Adam/actor/dense1/bias/v
*:(	2Adam/actor/means/kernel/v
#:!2Adam/actor/means/bias/v
.:,	2Adam/actor/log_stdss/kernel/v
':%2Adam/actor/log_stdss/bias/v
õ2ò
(__inference_actor_layer_call_fn_33364721Å
²
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
á2Þ
#__inference__wrapped_model_33364558¶
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
2
C__inference_actor_layer_call_and_return_conditional_losses_33364691Å
²
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
Ó2Ð
)__inference_dense1_layer_call_fn_33364801¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense1_layer_call_and_return_conditional_losses_33364812¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense2_layer_call_fn_33364821¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense2_layer_call_and_return_conditional_losses_33364832¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_means_layer_call_fn_33364841¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_means_layer_call_and_return_conditional_losses_33364851¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_log_stdss_layer_call_fn_33364860¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_log_stdss_layer_call_and_return_conditional_losses_33364870¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÍBÊ
&__inference_signature_wrapper_33364792input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
	J
Const
J	
Const_1ý
#__inference__wrapped_model_33364558Õ
#$Z[0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "ª
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿ
.
output_3"
output_3ÿÿÿÿÿÿÿÿÿò
C__inference_actor_layer_call_and_return_conditional_losses_33364691ª
#$Z[0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "j¢g
`¢]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 Ç
(__inference_actor_layer_call_fn_33364721
#$Z[0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "Z¢W

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿ¥
D__inference_dense1_layer_call_and_return_conditional_losses_33364812]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_dense1_layer_call_fn_33364801P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
D__inference_dense2_layer_call_and_return_conditional_losses_33364832^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense2_layer_call_fn_33364821Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
G__inference_log_stdss_layer_call_and_return_conditional_losses_33364870]#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_log_stdss_layer_call_fn_33364860P#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
C__inference_means_layer_call_and_return_conditional_losses_33364851]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_means_layer_call_fn_33364841P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_signature_wrapper_33364792à
#$Z[;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"ª
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿ
.
output_3"
output_3ÿÿÿÿÿÿÿÿÿ