û÷
³
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
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
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
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
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
H
ShardedFilename
basename	
shard

num_shards
filename
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718¢¯
·
-twinned_q_network_1/q_network_2/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	%*>
shared_name/-twinned_q_network_1/q_network_2/dense1/kernel
°
Atwinned_q_network_1/q_network_2/dense1/kernel/Read/ReadVariableOpReadVariableOp-twinned_q_network_1/q_network_2/dense1/kernel*
_output_shapes
:	%*
dtype0
¯
+twinned_q_network_1/q_network_2/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+twinned_q_network_1/q_network_2/dense1/bias
¨
?twinned_q_network_1/q_network_2/dense1/bias/Read/ReadVariableOpReadVariableOp+twinned_q_network_1/q_network_2/dense1/bias*
_output_shapes	
:*
dtype0
¸
-twinned_q_network_1/q_network_2/dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*>
shared_name/-twinned_q_network_1/q_network_2/dense2/kernel
±
Atwinned_q_network_1/q_network_2/dense2/kernel/Read/ReadVariableOpReadVariableOp-twinned_q_network_1/q_network_2/dense2/kernel* 
_output_shapes
:
*
dtype0
¯
+twinned_q_network_1/q_network_2/dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+twinned_q_network_1/q_network_2/dense2/bias
¨
?twinned_q_network_1/q_network_2/dense2/bias/Read/ReadVariableOpReadVariableOp+twinned_q_network_1/q_network_2/dense2/bias*
_output_shapes	
:*
dtype0
·
-twinned_q_network_1/q_network_2/output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*>
shared_name/-twinned_q_network_1/q_network_2/output/kernel
°
Atwinned_q_network_1/q_network_2/output/kernel/Read/ReadVariableOpReadVariableOp-twinned_q_network_1/q_network_2/output/kernel*
_output_shapes
:	*
dtype0
®
+twinned_q_network_1/q_network_2/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+twinned_q_network_1/q_network_2/output/bias
§
?twinned_q_network_1/q_network_2/output/bias/Read/ReadVariableOpReadVariableOp+twinned_q_network_1/q_network_2/output/bias*
_output_shapes
:*
dtype0
·
-twinned_q_network_1/q_network_3/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	%*>
shared_name/-twinned_q_network_1/q_network_3/dense1/kernel
°
Atwinned_q_network_1/q_network_3/dense1/kernel/Read/ReadVariableOpReadVariableOp-twinned_q_network_1/q_network_3/dense1/kernel*
_output_shapes
:	%*
dtype0
¯
+twinned_q_network_1/q_network_3/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+twinned_q_network_1/q_network_3/dense1/bias
¨
?twinned_q_network_1/q_network_3/dense1/bias/Read/ReadVariableOpReadVariableOp+twinned_q_network_1/q_network_3/dense1/bias*
_output_shapes	
:*
dtype0
¸
-twinned_q_network_1/q_network_3/dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*>
shared_name/-twinned_q_network_1/q_network_3/dense2/kernel
±
Atwinned_q_network_1/q_network_3/dense2/kernel/Read/ReadVariableOpReadVariableOp-twinned_q_network_1/q_network_3/dense2/kernel* 
_output_shapes
:
*
dtype0
¯
+twinned_q_network_1/q_network_3/dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+twinned_q_network_1/q_network_3/dense2/bias
¨
?twinned_q_network_1/q_network_3/dense2/bias/Read/ReadVariableOpReadVariableOp+twinned_q_network_1/q_network_3/dense2/bias*
_output_shapes	
:*
dtype0
·
-twinned_q_network_1/q_network_3/output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*>
shared_name/-twinned_q_network_1/q_network_3/output/kernel
°
Atwinned_q_network_1/q_network_3/output/kernel/Read/ReadVariableOpReadVariableOp-twinned_q_network_1/q_network_3/output/kernel*
_output_shapes
:	*
dtype0
®
+twinned_q_network_1/q_network_3/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+twinned_q_network_1/q_network_3/output/bias
§
?twinned_q_network_1/q_network_3/output/bias/Read/ReadVariableOpReadVariableOp+twinned_q_network_1/q_network_3/output/bias*
_output_shapes
:*
dtype0

NoOpNoOp
ß$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*$
value$B$ B$

Q1
Q2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures

	hidden_units


dense1

dense2
out
	variables
trainable_variables
regularization_losses
	keras_api

hidden_units

dense1

dense2
out
	variables
trainable_variables
regularization_losses
	keras_api
 
V
0
1
2
3
4
5
6
 7
!8
"9
#10
$11
V
0
1
2
3
4
5
6
 7
#8
$9
!10
"11
 
­
	variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
)non_trainable_variables
trainable_variables
regularization_losses
 
 
h

kernel
bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
h

kernel
bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
h

kernel
bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
*
0
1
2
3
4
5
*
0
1
2
3
4
5
 
­
	variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
:non_trainable_variables
trainable_variables
regularization_losses
 
h

kernel
 bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
h

!kernel
"bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
h

#kernel
$bias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
*
0
 1
!2
"3
#4
$5
*
0
 1
#2
$3
!4
"5
 
­
	variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
Knon_trainable_variables
trainable_variables
regularization_losses
ig
VARIABLE_VALUE-twinned_q_network_1/q_network_2/dense1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+twinned_q_network_1/q_network_2/dense1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE-twinned_q_network_1/q_network_2/dense2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+twinned_q_network_1/q_network_2/dense2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE-twinned_q_network_1/q_network_2/output/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+twinned_q_network_1/q_network_2/output/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE-twinned_q_network_1/q_network_3/dense1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+twinned_q_network_1/q_network_3/dense1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE-twinned_q_network_1/q_network_3/dense2/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+twinned_q_network_1/q_network_3/dense2/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-twinned_q_network_1/q_network_3/output/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+twinned_q_network_1/q_network_3/output/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 
 
 

0
1

0
1
 
­
*	variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
Pnon_trainable_variables
+trainable_variables
,regularization_losses

0
1

0
1
 
­
.	variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
Unon_trainable_variables
/trainable_variables
0regularization_losses

0
1

0
1
 
­
2	variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
Znon_trainable_variables
3trainable_variables
4regularization_losses


0
1
2
 
 
 
 

0
 1

0
 1
 
­
;	variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
_non_trainable_variables
<trainable_variables
=regularization_losses

!0
"1

!0
"1
 
­
?	variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
dnon_trainable_variables
@trainable_variables
Aregularization_losses

#0
$1

#0
$1
 
­
C	variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
inon_trainable_variables
Dtrainable_variables
Eregularization_losses

0
1
2
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
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ%

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1-twinned_q_network_1/q_network_2/dense1/kernel+twinned_q_network_1/q_network_2/dense1/bias-twinned_q_network_1/q_network_2/dense2/kernel+twinned_q_network_1/q_network_2/dense2/bias-twinned_q_network_1/q_network_2/output/kernel+twinned_q_network_1/q_network_2/output/bias-twinned_q_network_1/q_network_3/dense1/kernel+twinned_q_network_1/q_network_3/dense1/bias-twinned_q_network_1/q_network_3/dense2/kernel+twinned_q_network_1/q_network_3/dense2/bias-twinned_q_network_1/q_network_3/output/kernel+twinned_q_network_1/q_network_3/output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 */
f*R(
&__inference_signature_wrapper_23666084
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Æ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAtwinned_q_network_1/q_network_2/dense1/kernel/Read/ReadVariableOp?twinned_q_network_1/q_network_2/dense1/bias/Read/ReadVariableOpAtwinned_q_network_1/q_network_2/dense2/kernel/Read/ReadVariableOp?twinned_q_network_1/q_network_2/dense2/bias/Read/ReadVariableOpAtwinned_q_network_1/q_network_2/output/kernel/Read/ReadVariableOp?twinned_q_network_1/q_network_2/output/bias/Read/ReadVariableOpAtwinned_q_network_1/q_network_3/dense1/kernel/Read/ReadVariableOp?twinned_q_network_1/q_network_3/dense1/bias/Read/ReadVariableOpAtwinned_q_network_1/q_network_3/dense2/kernel/Read/ReadVariableOp?twinned_q_network_1/q_network_3/dense2/bias/Read/ReadVariableOpAtwinned_q_network_1/q_network_3/output/kernel/Read/ReadVariableOp?twinned_q_network_1/q_network_3/output/bias/Read/ReadVariableOpConst*
Tin
2*
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
!__inference__traced_save_23666262
Ñ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename-twinned_q_network_1/q_network_2/dense1/kernel+twinned_q_network_1/q_network_2/dense1/bias-twinned_q_network_1/q_network_2/dense2/kernel+twinned_q_network_1/q_network_2/dense2/bias-twinned_q_network_1/q_network_2/output/kernel+twinned_q_network_1/q_network_2/output/bias-twinned_q_network_1/q_network_3/dense1/kernel+twinned_q_network_1/q_network_3/dense1/bias-twinned_q_network_1/q_network_3/dense2/kernel+twinned_q_network_1/q_network_3/dense2/bias-twinned_q_network_1/q_network_3/output/kernel+twinned_q_network_1/q_network_3/output/bias*
Tin
2*
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
$__inference__traced_restore_23666308ñÐ
¤

)__inference_dense1_layer_call_fn_23666093

inputs
unknown:	%
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
D__inference_dense1_layer_call_and_return_conditional_losses_236657962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ%: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
 
_user_specified_nameinputs

¡
Q__inference_twinned_q_network_1_layer_call_and_return_conditional_losses_23666019
input_1'
q_network_2_23665991:	%#
q_network_2_23665993:	(
q_network_2_23665995:
#
q_network_2_23665997:	'
q_network_2_23665999:	"
q_network_2_23666001:'
q_network_3_23666004:	%#
q_network_3_23666006:	(
q_network_3_23666008:
#
q_network_3_23666010:	'
q_network_3_23666012:	"
q_network_3_23666014:
identity

identity_1¢#q_network_2/StatefulPartitionedCall¢#q_network_3/StatefulPartitionedCall
#q_network_2/StatefulPartitionedCallStatefulPartitionedCallinput_1q_network_2_23665991q_network_2_23665993q_network_2_23665995q_network_2_23665997q_network_2_23665999q_network_2_23666001*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_q_network_2_layer_call_and_return_conditional_losses_236658362%
#q_network_2/StatefulPartitionedCall
#q_network_3/StatefulPartitionedCallStatefulPartitionedCallinput_1q_network_3_23666004q_network_3_23666006q_network_3_23666008q_network_3_23666010q_network_3_23666012q_network_3_23666014*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_q_network_3_layer_call_and_return_conditional_losses_236659392%
#q_network_3/StatefulPartitionedCallÌ
IdentityIdentity,q_network_2/StatefulPartitionedCall:output:0$^q_network_2/StatefulPartitionedCall$^q_network_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÐ

Identity_1Identity,q_network_3/StatefulPartitionedCall:output:0$^q_network_2/StatefulPartitionedCall$^q_network_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ%: : : : : : : : : : : : 2J
#q_network_2/StatefulPartitionedCall#q_network_2/StatefulPartitionedCall2J
#q_network_3/StatefulPartitionedCall#q_network_3/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
!
_user_specified_name	input_1
Û

.__inference_q_network_3_layer_call_fn_23665957
input_1
unknown:	%
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_q_network_3_layer_call_and_return_conditional_losses_236659392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ%: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
!
_user_specified_name	input_1
Ô	
ö
D__inference_output_layer_call_and_return_conditional_losses_23666143

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
)__inference_dense1_layer_call_fn_23666152

inputs
unknown:	%
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
D__inference_dense1_layer_call_and_return_conditional_losses_236658992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ%: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
 
_user_specified_nameinputs
´

÷
D__inference_dense1_layer_call_and_return_conditional_losses_23665899

inputs1
matmul_readvariableop_resource:	%.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	%*
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
:ÿÿÿÿÿÿÿÿÿ%: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
 
_user_specified_nameinputs
´

÷
D__inference_dense1_layer_call_and_return_conditional_losses_23666104

inputs1
matmul_readvariableop_resource:	%.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	%*
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
:ÿÿÿÿÿÿÿÿÿ%: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
 
_user_specified_nameinputs


I__inference_q_network_2_layer_call_and_return_conditional_losses_23665836
input_1"
dense1_23665797:	%
dense1_23665799:	#
dense2_23665814:

dense2_23665816:	"
output_23665830:	
output_23665832:
identity¢dense1/StatefulPartitionedCall¢dense2/StatefulPartitionedCall¢output/StatefulPartitionedCall
dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_23665797dense1_23665799*
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
D__inference_dense1_layer_call_and_return_conditional_losses_236657962 
dense1/StatefulPartitionedCall·
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_23665814dense2_23665816*
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
D__inference_dense2_layer_call_and_return_conditional_losses_236658132 
dense2/StatefulPartitionedCall¶
output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_23665830output_23665832*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_236658292 
output/StatefulPartitionedCallÞ
IdentityIdentity'output/StatefulPartitionedCall:output:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ%: : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
!
_user_specified_name	input_1
¸

ø
D__inference_dense2_layer_call_and_return_conditional_losses_23665813

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
¸

ø
D__inference_dense2_layer_call_and_return_conditional_losses_23666124

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
´

÷
D__inference_dense1_layer_call_and_return_conditional_losses_23665796

inputs1
matmul_readvariableop_resource:	%.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	%*
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
:ÿÿÿÿÿÿÿÿÿ%: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
 
_user_specified_nameinputs
d
ó
#__inference__wrapped_model_23665781
input_1X
Etwinned_q_network_1_q_network_2_dense1_matmul_readvariableop_resource:	%U
Ftwinned_q_network_1_q_network_2_dense1_biasadd_readvariableop_resource:	Y
Etwinned_q_network_1_q_network_2_dense2_matmul_readvariableop_resource:
U
Ftwinned_q_network_1_q_network_2_dense2_biasadd_readvariableop_resource:	X
Etwinned_q_network_1_q_network_2_output_matmul_readvariableop_resource:	T
Ftwinned_q_network_1_q_network_2_output_biasadd_readvariableop_resource:X
Etwinned_q_network_1_q_network_3_dense1_matmul_readvariableop_resource:	%U
Ftwinned_q_network_1_q_network_3_dense1_biasadd_readvariableop_resource:	Y
Etwinned_q_network_1_q_network_3_dense2_matmul_readvariableop_resource:
U
Ftwinned_q_network_1_q_network_3_dense2_biasadd_readvariableop_resource:	X
Etwinned_q_network_1_q_network_3_output_matmul_readvariableop_resource:	T
Ftwinned_q_network_1_q_network_3_output_biasadd_readvariableop_resource:
identity

identity_1¢=twinned_q_network_1/q_network_2/dense1/BiasAdd/ReadVariableOp¢<twinned_q_network_1/q_network_2/dense1/MatMul/ReadVariableOp¢=twinned_q_network_1/q_network_2/dense2/BiasAdd/ReadVariableOp¢<twinned_q_network_1/q_network_2/dense2/MatMul/ReadVariableOp¢=twinned_q_network_1/q_network_2/output/BiasAdd/ReadVariableOp¢<twinned_q_network_1/q_network_2/output/MatMul/ReadVariableOp¢=twinned_q_network_1/q_network_3/dense1/BiasAdd/ReadVariableOp¢<twinned_q_network_1/q_network_3/dense1/MatMul/ReadVariableOp¢=twinned_q_network_1/q_network_3/dense2/BiasAdd/ReadVariableOp¢<twinned_q_network_1/q_network_3/dense2/MatMul/ReadVariableOp¢=twinned_q_network_1/q_network_3/output/BiasAdd/ReadVariableOp¢<twinned_q_network_1/q_network_3/output/MatMul/ReadVariableOp
<twinned_q_network_1/q_network_2/dense1/MatMul/ReadVariableOpReadVariableOpEtwinned_q_network_1_q_network_2_dense1_matmul_readvariableop_resource*
_output_shapes
:	%*
dtype02>
<twinned_q_network_1/q_network_2/dense1/MatMul/ReadVariableOpê
-twinned_q_network_1/q_network_2/dense1/MatMulMatMulinput_1Dtwinned_q_network_1/q_network_2/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-twinned_q_network_1/q_network_2/dense1/MatMul
=twinned_q_network_1/q_network_2/dense1/BiasAdd/ReadVariableOpReadVariableOpFtwinned_q_network_1_q_network_2_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02?
=twinned_q_network_1/q_network_2/dense1/BiasAdd/ReadVariableOp
.twinned_q_network_1/q_network_2/dense1/BiasAddBiasAdd7twinned_q_network_1/q_network_2/dense1/MatMul:product:0Etwinned_q_network_1/q_network_2/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.twinned_q_network_1/q_network_2/dense1/BiasAddÎ
+twinned_q_network_1/q_network_2/dense1/ReluRelu7twinned_q_network_1/q_network_2/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+twinned_q_network_1/q_network_2/dense1/Relu
<twinned_q_network_1/q_network_2/dense2/MatMul/ReadVariableOpReadVariableOpEtwinned_q_network_1_q_network_2_dense2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<twinned_q_network_1/q_network_2/dense2/MatMul/ReadVariableOp
-twinned_q_network_1/q_network_2/dense2/MatMulMatMul9twinned_q_network_1/q_network_2/dense1/Relu:activations:0Dtwinned_q_network_1/q_network_2/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-twinned_q_network_1/q_network_2/dense2/MatMul
=twinned_q_network_1/q_network_2/dense2/BiasAdd/ReadVariableOpReadVariableOpFtwinned_q_network_1_q_network_2_dense2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02?
=twinned_q_network_1/q_network_2/dense2/BiasAdd/ReadVariableOp
.twinned_q_network_1/q_network_2/dense2/BiasAddBiasAdd7twinned_q_network_1/q_network_2/dense2/MatMul:product:0Etwinned_q_network_1/q_network_2/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.twinned_q_network_1/q_network_2/dense2/BiasAddÎ
+twinned_q_network_1/q_network_2/dense2/ReluRelu7twinned_q_network_1/q_network_2/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+twinned_q_network_1/q_network_2/dense2/Relu
<twinned_q_network_1/q_network_2/output/MatMul/ReadVariableOpReadVariableOpEtwinned_q_network_1_q_network_2_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02>
<twinned_q_network_1/q_network_2/output/MatMul/ReadVariableOp
-twinned_q_network_1/q_network_2/output/MatMulMatMul9twinned_q_network_1/q_network_2/dense2/Relu:activations:0Dtwinned_q_network_1/q_network_2/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-twinned_q_network_1/q_network_2/output/MatMul
=twinned_q_network_1/q_network_2/output/BiasAdd/ReadVariableOpReadVariableOpFtwinned_q_network_1_q_network_2_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=twinned_q_network_1/q_network_2/output/BiasAdd/ReadVariableOp
.twinned_q_network_1/q_network_2/output/BiasAddBiasAdd7twinned_q_network_1/q_network_2/output/MatMul:product:0Etwinned_q_network_1/q_network_2/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.twinned_q_network_1/q_network_2/output/BiasAdd
<twinned_q_network_1/q_network_3/dense1/MatMul/ReadVariableOpReadVariableOpEtwinned_q_network_1_q_network_3_dense1_matmul_readvariableop_resource*
_output_shapes
:	%*
dtype02>
<twinned_q_network_1/q_network_3/dense1/MatMul/ReadVariableOpê
-twinned_q_network_1/q_network_3/dense1/MatMulMatMulinput_1Dtwinned_q_network_1/q_network_3/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-twinned_q_network_1/q_network_3/dense1/MatMul
=twinned_q_network_1/q_network_3/dense1/BiasAdd/ReadVariableOpReadVariableOpFtwinned_q_network_1_q_network_3_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02?
=twinned_q_network_1/q_network_3/dense1/BiasAdd/ReadVariableOp
.twinned_q_network_1/q_network_3/dense1/BiasAddBiasAdd7twinned_q_network_1/q_network_3/dense1/MatMul:product:0Etwinned_q_network_1/q_network_3/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.twinned_q_network_1/q_network_3/dense1/BiasAddÎ
+twinned_q_network_1/q_network_3/dense1/ReluRelu7twinned_q_network_1/q_network_3/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+twinned_q_network_1/q_network_3/dense1/Relu
<twinned_q_network_1/q_network_3/dense2/MatMul/ReadVariableOpReadVariableOpEtwinned_q_network_1_q_network_3_dense2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<twinned_q_network_1/q_network_3/dense2/MatMul/ReadVariableOp
-twinned_q_network_1/q_network_3/dense2/MatMulMatMul9twinned_q_network_1/q_network_3/dense1/Relu:activations:0Dtwinned_q_network_1/q_network_3/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-twinned_q_network_1/q_network_3/dense2/MatMul
=twinned_q_network_1/q_network_3/dense2/BiasAdd/ReadVariableOpReadVariableOpFtwinned_q_network_1_q_network_3_dense2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02?
=twinned_q_network_1/q_network_3/dense2/BiasAdd/ReadVariableOp
.twinned_q_network_1/q_network_3/dense2/BiasAddBiasAdd7twinned_q_network_1/q_network_3/dense2/MatMul:product:0Etwinned_q_network_1/q_network_3/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.twinned_q_network_1/q_network_3/dense2/BiasAddÎ
+twinned_q_network_1/q_network_3/dense2/ReluRelu7twinned_q_network_1/q_network_3/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+twinned_q_network_1/q_network_3/dense2/Relu
<twinned_q_network_1/q_network_3/output/MatMul/ReadVariableOpReadVariableOpEtwinned_q_network_1_q_network_3_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02>
<twinned_q_network_1/q_network_3/output/MatMul/ReadVariableOp
-twinned_q_network_1/q_network_3/output/MatMulMatMul9twinned_q_network_1/q_network_3/dense2/Relu:activations:0Dtwinned_q_network_1/q_network_3/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-twinned_q_network_1/q_network_3/output/MatMul
=twinned_q_network_1/q_network_3/output/BiasAdd/ReadVariableOpReadVariableOpFtwinned_q_network_1_q_network_3_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=twinned_q_network_1/q_network_3/output/BiasAdd/ReadVariableOp
.twinned_q_network_1/q_network_3/output/BiasAddBiasAdd7twinned_q_network_1/q_network_3/output/MatMul:product:0Etwinned_q_network_1/q_network_3/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.twinned_q_network_1/q_network_3/output/BiasAdd
IdentityIdentity7twinned_q_network_1/q_network_2/output/BiasAdd:output:0>^twinned_q_network_1/q_network_2/dense1/BiasAdd/ReadVariableOp=^twinned_q_network_1/q_network_2/dense1/MatMul/ReadVariableOp>^twinned_q_network_1/q_network_2/dense2/BiasAdd/ReadVariableOp=^twinned_q_network_1/q_network_2/dense2/MatMul/ReadVariableOp>^twinned_q_network_1/q_network_2/output/BiasAdd/ReadVariableOp=^twinned_q_network_1/q_network_2/output/MatMul/ReadVariableOp>^twinned_q_network_1/q_network_3/dense1/BiasAdd/ReadVariableOp=^twinned_q_network_1/q_network_3/dense1/MatMul/ReadVariableOp>^twinned_q_network_1/q_network_3/dense2/BiasAdd/ReadVariableOp=^twinned_q_network_1/q_network_3/dense2/MatMul/ReadVariableOp>^twinned_q_network_1/q_network_3/output/BiasAdd/ReadVariableOp=^twinned_q_network_1/q_network_3/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity7twinned_q_network_1/q_network_3/output/BiasAdd:output:0>^twinned_q_network_1/q_network_2/dense1/BiasAdd/ReadVariableOp=^twinned_q_network_1/q_network_2/dense1/MatMul/ReadVariableOp>^twinned_q_network_1/q_network_2/dense2/BiasAdd/ReadVariableOp=^twinned_q_network_1/q_network_2/dense2/MatMul/ReadVariableOp>^twinned_q_network_1/q_network_2/output/BiasAdd/ReadVariableOp=^twinned_q_network_1/q_network_2/output/MatMul/ReadVariableOp>^twinned_q_network_1/q_network_3/dense1/BiasAdd/ReadVariableOp=^twinned_q_network_1/q_network_3/dense1/MatMul/ReadVariableOp>^twinned_q_network_1/q_network_3/dense2/BiasAdd/ReadVariableOp=^twinned_q_network_1/q_network_3/dense2/MatMul/ReadVariableOp>^twinned_q_network_1/q_network_3/output/BiasAdd/ReadVariableOp=^twinned_q_network_1/q_network_3/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ%: : : : : : : : : : : : 2~
=twinned_q_network_1/q_network_2/dense1/BiasAdd/ReadVariableOp=twinned_q_network_1/q_network_2/dense1/BiasAdd/ReadVariableOp2|
<twinned_q_network_1/q_network_2/dense1/MatMul/ReadVariableOp<twinned_q_network_1/q_network_2/dense1/MatMul/ReadVariableOp2~
=twinned_q_network_1/q_network_2/dense2/BiasAdd/ReadVariableOp=twinned_q_network_1/q_network_2/dense2/BiasAdd/ReadVariableOp2|
<twinned_q_network_1/q_network_2/dense2/MatMul/ReadVariableOp<twinned_q_network_1/q_network_2/dense2/MatMul/ReadVariableOp2~
=twinned_q_network_1/q_network_2/output/BiasAdd/ReadVariableOp=twinned_q_network_1/q_network_2/output/BiasAdd/ReadVariableOp2|
<twinned_q_network_1/q_network_2/output/MatMul/ReadVariableOp<twinned_q_network_1/q_network_2/output/MatMul/ReadVariableOp2~
=twinned_q_network_1/q_network_3/dense1/BiasAdd/ReadVariableOp=twinned_q_network_1/q_network_3/dense1/BiasAdd/ReadVariableOp2|
<twinned_q_network_1/q_network_3/dense1/MatMul/ReadVariableOp<twinned_q_network_1/q_network_3/dense1/MatMul/ReadVariableOp2~
=twinned_q_network_1/q_network_3/dense2/BiasAdd/ReadVariableOp=twinned_q_network_1/q_network_3/dense2/BiasAdd/ReadVariableOp2|
<twinned_q_network_1/q_network_3/dense2/MatMul/ReadVariableOp<twinned_q_network_1/q_network_3/dense2/MatMul/ReadVariableOp2~
=twinned_q_network_1/q_network_3/output/BiasAdd/ReadVariableOp=twinned_q_network_1/q_network_3/output/BiasAdd/ReadVariableOp2|
<twinned_q_network_1/q_network_3/output/MatMul/ReadVariableOp<twinned_q_network_1/q_network_3/output/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
!
_user_specified_name	input_1


I__inference_q_network_3_layer_call_and_return_conditional_losses_23665939
input_1"
dense1_23665900:	%
dense1_23665902:	#
dense2_23665917:

dense2_23665919:	"
output_23665933:	
output_23665935:
identity¢dense1/StatefulPartitionedCall¢dense2/StatefulPartitionedCall¢output/StatefulPartitionedCall
dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_23665900dense1_23665902*
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
D__inference_dense1_layer_call_and_return_conditional_losses_236658992 
dense1/StatefulPartitionedCall·
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_23665917dense2_23665919*
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
D__inference_dense2_layer_call_and_return_conditional_losses_236659162 
dense2/StatefulPartitionedCall¶
output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_23665933output_23665935*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_236659322 
output/StatefulPartitionedCallÞ
IdentityIdentity'output/StatefulPartitionedCall:output:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ%: : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
!
_user_specified_name	input_1
¸

ø
D__inference_dense2_layer_call_and_return_conditional_losses_23665916

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
Ô	
ö
D__inference_output_layer_call_and_return_conditional_losses_23665932

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
D__inference_dense2_layer_call_and_return_conditional_losses_23666183

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
´

÷
D__inference_dense1_layer_call_and_return_conditional_losses_23666163

inputs1
matmul_readvariableop_resource:	%.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	%*
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
:ÿÿÿÿÿÿÿÿÿ%: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
 
_user_specified_nameinputs

¿
&__inference_signature_wrapper_23666084
input_1
unknown:	%
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	%
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity

identity_1¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *,
f'R%
#__inference__wrapped_model_236657812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ%: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
!
_user_specified_name	input_1
§

)__inference_dense2_layer_call_fn_23666113

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
D__inference_dense2_layer_call_and_return_conditional_losses_236658132
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
Ô	
ö
D__inference_output_layer_call_and_return_conditional_losses_23666202

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
Ë
Ï
6__inference_twinned_q_network_1_layer_call_fn_23666051
input_1
unknown:	%
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	%
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_twinned_q_network_1_layer_call_and_return_conditional_losses_236660192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ%: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
!
_user_specified_name	input_1
£

)__inference_output_layer_call_fn_23666133

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_236658292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
Ô	
ö
D__inference_output_layer_call_and_return_conditional_losses_23665829

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
Ë:


$__inference__traced_restore_23666308
file_prefixQ
>assignvariableop_twinned_q_network_1_q_network_2_dense1_kernel:	%M
>assignvariableop_1_twinned_q_network_1_q_network_2_dense1_bias:	T
@assignvariableop_2_twinned_q_network_1_q_network_2_dense2_kernel:
M
>assignvariableop_3_twinned_q_network_1_q_network_2_dense2_bias:	S
@assignvariableop_4_twinned_q_network_1_q_network_2_output_kernel:	L
>assignvariableop_5_twinned_q_network_1_q_network_2_output_bias:S
@assignvariableop_6_twinned_q_network_1_q_network_3_dense1_kernel:	%M
>assignvariableop_7_twinned_q_network_1_q_network_3_dense1_bias:	T
@assignvariableop_8_twinned_q_network_1_q_network_3_dense2_kernel:
M
>assignvariableop_9_twinned_q_network_1_q_network_3_dense2_bias:	T
Aassignvariableop_10_twinned_q_network_1_q_network_3_output_kernel:	M
?assignvariableop_11_twinned_q_network_1_q_network_3_output_bias:
identity_13¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¨
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesì
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity½
AssignVariableOpAssignVariableOp>assignvariableop_twinned_q_network_1_q_network_2_dense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ã
AssignVariableOp_1AssignVariableOp>assignvariableop_1_twinned_q_network_1_q_network_2_dense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Å
AssignVariableOp_2AssignVariableOp@assignvariableop_2_twinned_q_network_1_q_network_2_dense2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ã
AssignVariableOp_3AssignVariableOp>assignvariableop_3_twinned_q_network_1_q_network_2_dense2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Å
AssignVariableOp_4AssignVariableOp@assignvariableop_4_twinned_q_network_1_q_network_2_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ã
AssignVariableOp_5AssignVariableOp>assignvariableop_5_twinned_q_network_1_q_network_2_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Å
AssignVariableOp_6AssignVariableOp@assignvariableop_6_twinned_q_network_1_q_network_3_dense1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ã
AssignVariableOp_7AssignVariableOp>assignvariableop_7_twinned_q_network_1_q_network_3_dense1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Å
AssignVariableOp_8AssignVariableOp@assignvariableop_8_twinned_q_network_1_q_network_3_dense2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ã
AssignVariableOp_9AssignVariableOp>assignvariableop_9_twinned_q_network_1_q_network_3_dense2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10É
AssignVariableOp_10AssignVariableOpAassignvariableop_10_twinned_q_network_1_q_network_3_output_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ç
AssignVariableOp_11AssignVariableOp?assignvariableop_11_twinned_q_network_1_q_network_3_output_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpæ
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12Ù
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
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
Û

.__inference_q_network_2_layer_call_fn_23665854
input_1
unknown:	%
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_q_network_2_layer_call_and_return_conditional_losses_236658362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ%: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
!
_user_specified_name	input_1
£

)__inference_output_layer_call_fn_23666192

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_236659322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
)

!__inference__traced_save_23666262
file_prefixL
Hsavev2_twinned_q_network_1_q_network_2_dense1_kernel_read_readvariableopJ
Fsavev2_twinned_q_network_1_q_network_2_dense1_bias_read_readvariableopL
Hsavev2_twinned_q_network_1_q_network_2_dense2_kernel_read_readvariableopJ
Fsavev2_twinned_q_network_1_q_network_2_dense2_bias_read_readvariableopL
Hsavev2_twinned_q_network_1_q_network_2_output_kernel_read_readvariableopJ
Fsavev2_twinned_q_network_1_q_network_2_output_bias_read_readvariableopL
Hsavev2_twinned_q_network_1_q_network_3_dense1_kernel_read_readvariableopJ
Fsavev2_twinned_q_network_1_q_network_3_dense1_bias_read_readvariableopL
Hsavev2_twinned_q_network_1_q_network_3_dense2_kernel_read_readvariableopJ
Fsavev2_twinned_q_network_1_q_network_3_dense2_bias_read_readvariableopL
Hsavev2_twinned_q_network_1_q_network_3_output_kernel_read_readvariableopJ
Fsavev2_twinned_q_network_1_q_network_3_output_bias_read_readvariableop
savev2_const

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
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¢
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices²
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Hsavev2_twinned_q_network_1_q_network_2_dense1_kernel_read_readvariableopFsavev2_twinned_q_network_1_q_network_2_dense1_bias_read_readvariableopHsavev2_twinned_q_network_1_q_network_2_dense2_kernel_read_readvariableopFsavev2_twinned_q_network_1_q_network_2_dense2_bias_read_readvariableopHsavev2_twinned_q_network_1_q_network_2_output_kernel_read_readvariableopFsavev2_twinned_q_network_1_q_network_2_output_bias_read_readvariableopHsavev2_twinned_q_network_1_q_network_3_dense1_kernel_read_readvariableopFsavev2_twinned_q_network_1_q_network_3_dense1_bias_read_readvariableopHsavev2_twinned_q_network_1_q_network_3_dense2_kernel_read_readvariableopFsavev2_twinned_q_network_1_q_network_3_dense2_bias_read_readvariableopHsavev2_twinned_q_network_1_q_network_3_output_kernel_read_readvariableopFsavev2_twinned_q_network_1_q_network_3_output_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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

identity_1Identity_1:output:0*
_input_shapesr
p: :	%::
::	::	%::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	%:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	%:!

_output_shapes	
::&	"
 
_output_shapes
:
:!


_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: 
§

)__inference_dense2_layer_call_fn_23666172

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
D__inference_dense2_layer_call_and_return_conditional_losses_236659162
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
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*é
serving_defaultÕ
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ%<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ<
output_20
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¸

Q1
Q2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
j__call__
k_default_save_signature
*l&call_and_return_all_conditional_losses"±
_tf_keras_model{"name": "twinned_q_network_1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "TwinnedQNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 37]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "TwinnedQNetwork"}}
Ý
	hidden_units


dense1

dense2
out
	variables
trainable_variables
regularization_losses
	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_model{"name": "q_network_2", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "QNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 37]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "QNetwork"}}
Ý
hidden_units

dense1

dense2
out
	variables
trainable_variables
regularization_losses
	keras_api
o__call__
*p&call_and_return_all_conditional_losses"
_tf_keras_model{"name": "q_network_3", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "QNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 37]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "QNetwork"}}
"
	optimizer
v
0
1
2
3
4
5
6
 7
!8
"9
#10
$11"
trackable_list_wrapper
v
0
1
2
3
4
5
6
 7
#8
$9
!10
"11"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
	variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
)non_trainable_variables
trainable_variables
regularization_losses
j__call__
k_default_save_signature
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
,
qserving_default"
signature_map
 "
trackable_list_wrapper
È

kernel
bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
r__call__
*s&call_and_return_all_conditional_losses"£
_tf_keras_layer{"name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 37}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 37]}}
Ê

kernel
bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
t__call__
*u&call_and_return_all_conditional_losses"¥
_tf_keras_layer{"name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
Ì

kernel
bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
v__call__
*w&call_and_return_all_conditional_losses"§
_tf_keras_layer{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float64", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
:non_trainable_variables
trainable_variables
regularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
Ì

kernel
 bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
x__call__
*y&call_and_return_all_conditional_losses"§
_tf_keras_layer{"name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 37}}, "shared_object_id": 15}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 37]}}
Î

!kernel
"bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
z__call__
*{&call_and_return_all_conditional_losses"©
_tf_keras_layer{"name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
Î

#kernel
$bias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
|__call__
*}&call_and_return_all_conditional_losses"©
_tf_keras_layer{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float64", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
J
0
 1
!2
"3
#4
$5"
trackable_list_wrapper
J
0
 1
#2
$3
!4
"5"
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
Knon_trainable_variables
trainable_variables
regularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
@:>	%2-twinned_q_network_1/q_network_2/dense1/kernel
::82+twinned_q_network_1/q_network_2/dense1/bias
A:?
2-twinned_q_network_1/q_network_2/dense2/kernel
::82+twinned_q_network_1/q_network_2/dense2/bias
@:>	2-twinned_q_network_1/q_network_2/output/kernel
9:72+twinned_q_network_1/q_network_2/output/bias
@:>	%2-twinned_q_network_1/q_network_3/dense1/kernel
::82+twinned_q_network_1/q_network_3/dense1/bias
A:?
2-twinned_q_network_1/q_network_3/dense2/kernel
::82+twinned_q_network_1/q_network_3/dense2/bias
@:>	2-twinned_q_network_1/q_network_3/output/kernel
9:72+twinned_q_network_1/q_network_3/output/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
*	variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
Pnon_trainable_variables
+trainable_variables
,regularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
.	variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
Unon_trainable_variables
/trainable_variables
0regularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
2	variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
Znon_trainable_variables
3trainable_variables
4regularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
5

0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
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
­
;	variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
_non_trainable_variables
<trainable_variables
=regularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
?	variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
dnon_trainable_variables
@trainable_variables
Aregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
C	variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
inon_trainable_variables
Dtrainable_variables
Eregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
5
0
1
2"
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
ÿ2ü
6__inference_twinned_q_network_1_layer_call_fn_23666051Á
²
FullArgSpec
args
jself
jx
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
input_1ÿÿÿÿÿÿÿÿÿ%
á2Þ
#__inference__wrapped_model_23665781¶
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
input_1ÿÿÿÿÿÿÿÿÿ%
2
Q__inference_twinned_q_network_1_layer_call_and_return_conditional_losses_23666019Á
²
FullArgSpec
args
jself
jx
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
input_1ÿÿÿÿÿÿÿÿÿ%
÷2ô
.__inference_q_network_2_layer_call_fn_23665854Á
²
FullArgSpec
args
jself
jx
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
input_1ÿÿÿÿÿÿÿÿÿ%
2
I__inference_q_network_2_layer_call_and_return_conditional_losses_23665836Á
²
FullArgSpec
args
jself
jx
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
input_1ÿÿÿÿÿÿÿÿÿ%
÷2ô
.__inference_q_network_3_layer_call_fn_23665957Á
²
FullArgSpec
args
jself
jx
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
input_1ÿÿÿÿÿÿÿÿÿ%
2
I__inference_q_network_3_layer_call_and_return_conditional_losses_23665939Á
²
FullArgSpec
args
jself
jx
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
input_1ÿÿÿÿÿÿÿÿÿ%
ÍBÊ
&__inference_signature_wrapper_23666084input_1"
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
Ó2Ð
)__inference_dense1_layer_call_fn_23666093¢
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
D__inference_dense1_layer_call_and_return_conditional_losses_23666104¢
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
)__inference_dense2_layer_call_fn_23666113¢
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
D__inference_dense2_layer_call_and_return_conditional_losses_23666124¢
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
)__inference_output_layer_call_fn_23666133¢
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
D__inference_output_layer_call_and_return_conditional_losses_23666143¢
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
)__inference_dense1_layer_call_fn_23666152¢
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
D__inference_dense1_layer_call_and_return_conditional_losses_23666163¢
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
)__inference_dense2_layer_call_fn_23666172¢
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
D__inference_dense2_layer_call_and_return_conditional_losses_23666183¢
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
)__inference_output_layer_call_fn_23666192¢
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
D__inference_output_layer_call_and_return_conditional_losses_23666202¢
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
 Í
#__inference__wrapped_model_23665781¥ !"#$0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ%
ª "cª`
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿ¥
D__inference_dense1_layer_call_and_return_conditional_losses_23666104]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ%
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¥
D__inference_dense1_layer_call_and_return_conditional_losses_23666163] /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ%
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_dense1_layer_call_fn_23666093P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ%
ª "ÿÿÿÿÿÿÿÿÿ}
)__inference_dense1_layer_call_fn_23666152P /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ%
ª "ÿÿÿÿÿÿÿÿÿ¦
D__inference_dense2_layer_call_and_return_conditional_losses_23666124^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¦
D__inference_dense2_layer_call_and_return_conditional_losses_23666183^!"0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense2_layer_call_fn_23666113Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ~
)__inference_dense2_layer_call_fn_23666172Q!"0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
D__inference_output_layer_call_and_return_conditional_losses_23666143]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¥
D__inference_output_layer_call_and_return_conditional_losses_23666202]#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_output_layer_call_fn_23666133P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ}
)__inference_output_layer_call_fn_23666192P#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ®
I__inference_q_network_2_layer_call_and_return_conditional_losses_23665836a0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ%
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_q_network_2_layer_call_fn_23665854T0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ%
ª "ÿÿÿÿÿÿÿÿÿ®
I__inference_q_network_3_layer_call_and_return_conditional_losses_23665939a !"#$0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ%
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_q_network_3_layer_call_fn_23665957T !"#$0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ%
ª "ÿÿÿÿÿÿÿÿÿÛ
&__inference_signature_wrapper_23666084° !"#$;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ%"cª`
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿã
Q__inference_twinned_q_network_1_layer_call_and_return_conditional_losses_23666019 !"#$0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ%
ª "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 ¹
6__inference_twinned_q_network_1_layer_call_fn_23666051 !"#$0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ%
ª "=¢:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ