¿ö
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
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718¯
·
-twinned_q_network_1/q_network_2/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*>
shared_name/-twinned_q_network_1/q_network_2/dense1/kernel
°
Atwinned_q_network_1/q_network_2/dense1/kernel/Read/ReadVariableOpReadVariableOp-twinned_q_network_1/q_network_2/dense1/kernel*
_output_shapes
:	*
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
-twinned_q_network_1/q_network_3/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*>
shared_name/-twinned_q_network_1/q_network_3/dense1/kernel
°
Atwinned_q_network_1/q_network_3/dense1/kernel/Read/ReadVariableOpReadVariableOp-twinned_q_network_1/q_network_3/dense1/kernel*
_output_shapes
:	*
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

NoOpNoOp
×%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*%
value%B% Bþ$

Q1
Q2
	optimizer
trainable_variables
	variables
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
trainable_variables
	variables
regularization_losses
	keras_api

hidden_units

dense1

dense2
out
trainable_variables
	variables
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
%non_trainable_variables

&layers
trainable_variables
'layer_regularization_losses
	variables
regularization_losses
(metrics
)layer_metrics
 
 
h

kernel
bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
h

kernel
bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
h

kernel
bias
2trainable_variables
3	variables
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
6non_trainable_variables

7layers
trainable_variables
8layer_regularization_losses
	variables
regularization_losses
9metrics
:layer_metrics
 
h

kernel
 bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
h

#kernel
$bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
h

!kernel
"bias
Ctrainable_variables
D	variables
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
Gnon_trainable_variables

Hlayers
trainable_variables
Ilayer_regularization_losses
	variables
regularization_losses
Jmetrics
Klayer_metrics
sq
VARIABLE_VALUE-twinned_q_network_1/q_network_2/dense1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE+twinned_q_network_1/q_network_2/dense1/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE-twinned_q_network_1/q_network_2/output/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE+twinned_q_network_1/q_network_2/output/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE-twinned_q_network_1/q_network_2/dense2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE+twinned_q_network_1/q_network_2/dense2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE-twinned_q_network_1/q_network_3/dense1/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE+twinned_q_network_1/q_network_3/dense1/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE-twinned_q_network_1/q_network_3/output/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE+twinned_q_network_1/q_network_3/output/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE-twinned_q_network_1/q_network_3/dense2/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE+twinned_q_network_1/q_network_3/dense2/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
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
Lnon_trainable_variables

Mlayers
*trainable_variables
Nlayer_regularization_losses
+	variables
,regularization_losses
Ometrics
Player_metrics

0
1

0
1
 
­
Qnon_trainable_variables

Rlayers
.trainable_variables
Slayer_regularization_losses
/	variables
0regularization_losses
Tmetrics
Ulayer_metrics

0
1

0
1
 
­
Vnon_trainable_variables

Wlayers
2trainable_variables
Xlayer_regularization_losses
3	variables
4regularization_losses
Ymetrics
Zlayer_metrics
 


0
1
2
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
[non_trainable_variables

\layers
;trainable_variables
]layer_regularization_losses
<	variables
=regularization_losses
^metrics
_layer_metrics

#0
$1

#0
$1
 
­
`non_trainable_variables

alayers
?trainable_variables
blayer_regularization_losses
@	variables
Aregularization_losses
cmetrics
dlayer_metrics

!0
"1

!0
"1
 
­
enon_trainable_variables

flayers
Ctrainable_variables
glayer_regularization_losses
D	variables
Eregularization_losses
hmetrics
ilayer_metrics
 
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
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

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
GPU2 *0J 8 *+
f&R$
"__inference_signature_wrapper_3365
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Â
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAtwinned_q_network_1/q_network_2/dense1/kernel/Read/ReadVariableOp?twinned_q_network_1/q_network_2/dense1/bias/Read/ReadVariableOpAtwinned_q_network_1/q_network_2/output/kernel/Read/ReadVariableOp?twinned_q_network_1/q_network_2/output/bias/Read/ReadVariableOpAtwinned_q_network_1/q_network_2/dense2/kernel/Read/ReadVariableOp?twinned_q_network_1/q_network_2/dense2/bias/Read/ReadVariableOpAtwinned_q_network_1/q_network_3/dense1/kernel/Read/ReadVariableOp?twinned_q_network_1/q_network_3/dense1/bias/Read/ReadVariableOpAtwinned_q_network_1/q_network_3/output/kernel/Read/ReadVariableOp?twinned_q_network_1/q_network_3/output/bias/Read/ReadVariableOpAtwinned_q_network_1/q_network_3/dense2/kernel/Read/ReadVariableOp?twinned_q_network_1/q_network_3/dense2/bias/Read/ReadVariableOpConst*
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
GPU2 *0J 8 *&
f!R
__inference__traced_save_3543
Í
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename-twinned_q_network_1/q_network_2/dense1/kernel+twinned_q_network_1/q_network_2/dense1/bias-twinned_q_network_1/q_network_2/output/kernel+twinned_q_network_1/q_network_2/output/bias-twinned_q_network_1/q_network_2/dense2/kernel+twinned_q_network_1/q_network_2/dense2/bias-twinned_q_network_1/q_network_3/dense1/kernel+twinned_q_network_1/q_network_3/dense1/bias-twinned_q_network_1/q_network_3/output/kernel+twinned_q_network_1/q_network_3/output/bias-twinned_q_network_1/q_network_3/dense2/kernel+twinned_q_network_1/q_network_3/dense2/bias*
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
GPU2 *0J 8 *)
f$R"
 __inference__traced_restore_3589éÏ
´

ô
@__inference_dense2_layer_call_and_return_conditional_losses_3464

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
Ð	
ò
@__inference_output_layer_call_and_return_conditional_losses_3213

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


%__inference_dense2_layer_call_fn_3453

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallö
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
GPU2 *0J 8 *I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_31972
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
Ð	
ò
@__inference_output_layer_call_and_return_conditional_losses_3424

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
Ð	
ò
@__inference_output_layer_call_and_return_conditional_losses_3483

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


%__inference_dense1_layer_call_fn_3433

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallö
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
GPU2 *0J 8 *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_31802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó

*__inference_q_network_2_layer_call_fn_3135
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
identity¢StatefulPartitionedCall¯
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
GPU2 *0J 8 *N
fIRG
E__inference_q_network_2_layer_call_and_return_conditional_losses_31172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1


%__inference_dense2_layer_call_fn_3394

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallö
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
GPU2 *0J 8 *I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_30942
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
¿;


 __inference__traced_restore_3589
file_prefixQ
>assignvariableop_twinned_q_network_1_q_network_2_dense1_kernel:	M
>assignvariableop_1_twinned_q_network_1_q_network_2_dense1_bias:	S
@assignvariableop_2_twinned_q_network_1_q_network_2_output_kernel:	L
>assignvariableop_3_twinned_q_network_1_q_network_2_output_bias:T
@assignvariableop_4_twinned_q_network_1_q_network_2_dense2_kernel:
M
>assignvariableop_5_twinned_q_network_1_q_network_2_dense2_bias:	S
@assignvariableop_6_twinned_q_network_1_q_network_3_dense1_kernel:	M
>assignvariableop_7_twinned_q_network_1_q_network_3_dense1_bias:	S
@assignvariableop_8_twinned_q_network_1_q_network_3_output_kernel:	L
>assignvariableop_9_twinned_q_network_1_q_network_3_output_bias:U
Aassignvariableop_10_twinned_q_network_1_q_network_3_dense2_kernel:
N
?assignvariableop_11_twinned_q_network_1_q_network_3_dense2_bias:	
identity_13¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOp_2AssignVariableOp@assignvariableop_2_twinned_q_network_1_q_network_2_output_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ã
AssignVariableOp_3AssignVariableOp>assignvariableop_3_twinned_q_network_1_q_network_2_output_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Å
AssignVariableOp_4AssignVariableOp@assignvariableop_4_twinned_q_network_1_q_network_2_dense2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ã
AssignVariableOp_5AssignVariableOp>assignvariableop_5_twinned_q_network_1_q_network_2_dense2_biasIdentity_5:output:0"/device:CPU:0*
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
AssignVariableOp_8AssignVariableOp@assignvariableop_8_twinned_q_network_1_q_network_3_output_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ã
AssignVariableOp_9AssignVariableOp>assignvariableop_9_twinned_q_network_1_q_network_3_output_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10É
AssignVariableOp_10AssignVariableOpAassignvariableop_10_twinned_q_network_1_q_network_3_dense2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ç
AssignVariableOp_11AssignVariableOp?assignvariableop_11_twinned_q_network_1_q_network_3_dense2_biasIdentity_11:output:0"/device:CPU:0*
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
Ó

*__inference_q_network_3_layer_call_fn_3238
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
identity¢StatefulPartitionedCall¯
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
GPU2 *0J 8 *N
fIRG
E__inference_q_network_3_layer_call_and_return_conditional_losses_32202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Þ
ü
E__inference_q_network_3_layer_call_and_return_conditional_losses_3220
input_1
dense1_3181:	
dense1_3183:	
dense2_3198:

dense2_3200:	
output_3214:	
output_3216:
identity¢dense1/StatefulPartitionedCall¢dense2/StatefulPartitionedCall¢output/StatefulPartitionedCall
dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_3181dense1_3183*
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
GPU2 *0J 8 *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_31802 
dense1/StatefulPartitionedCall«
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_3198dense2_3200*
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
GPU2 *0J 8 *I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_31972 
dense2/StatefulPartitionedCallª
output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_3214output_3216*
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
GPU2 *0J 8 *I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_32132 
output/StatefulPartitionedCallÞ
IdentityIdentity'output/StatefulPartitionedCall:output:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Þ
ü
E__inference_q_network_2_layer_call_and_return_conditional_losses_3117
input_1
dense1_3078:	
dense1_3080:	
dense2_3095:

dense2_3097:	
output_3111:	
output_3113:
identity¢dense1/StatefulPartitionedCall¢dense2/StatefulPartitionedCall¢output/StatefulPartitionedCall
dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_3078dense1_3080*
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
GPU2 *0J 8 *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_30772 
dense1/StatefulPartitionedCall«
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_3095dense2_3097*
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
GPU2 *0J 8 *I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_30942 
dense2/StatefulPartitionedCallª
output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_3111output_3113*
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
GPU2 *0J 8 *I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_31102 
output/StatefulPartitionedCallÞ
IdentityIdentity'output/StatefulPartitionedCall:output:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

í
M__inference_twinned_q_network_1_layer_call_and_return_conditional_losses_3300
input_1#
q_network_2_3272:	
q_network_2_3274:	$
q_network_2_3276:

q_network_2_3278:	#
q_network_2_3280:	
q_network_2_3282:#
q_network_3_3285:	
q_network_3_3287:	$
q_network_3_3289:

q_network_3_3291:	#
q_network_3_3293:	
q_network_3_3295:
identity

identity_1¢#q_network_2/StatefulPartitionedCall¢#q_network_3/StatefulPartitionedCalló
#q_network_2/StatefulPartitionedCallStatefulPartitionedCallinput_1q_network_2_3272q_network_2_3274q_network_2_3276q_network_2_3278q_network_2_3280q_network_2_3282*
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
GPU2 *0J 8 *N
fIRG
E__inference_q_network_2_layer_call_and_return_conditional_losses_31172%
#q_network_2/StatefulPartitionedCalló
#q_network_3/StatefulPartitionedCallStatefulPartitionedCallinput_1q_network_3_3285q_network_3_3287q_network_3_3289q_network_3_3291q_network_3_3293q_network_3_3295*
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
GPU2 *0J 8 *N
fIRG
E__inference_q_network_3_layer_call_and_return_conditional_losses_32202%
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
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2J
#q_network_2/StatefulPartitionedCall#q_network_2/StatefulPartitionedCall2J
#q_network_3/StatefulPartitionedCall#q_network_3/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ð	
ò
@__inference_output_layer_call_and_return_conditional_losses_3110

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
°

ó
@__inference_dense1_layer_call_and_return_conditional_losses_3385

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
*

__inference__traced_save_3543
file_prefixL
Hsavev2_twinned_q_network_1_q_network_2_dense1_kernel_read_readvariableopJ
Fsavev2_twinned_q_network_1_q_network_2_dense1_bias_read_readvariableopL
Hsavev2_twinned_q_network_1_q_network_2_output_kernel_read_readvariableopJ
Fsavev2_twinned_q_network_1_q_network_2_output_bias_read_readvariableopL
Hsavev2_twinned_q_network_1_q_network_2_dense2_kernel_read_readvariableopJ
Fsavev2_twinned_q_network_1_q_network_2_dense2_bias_read_readvariableopL
Hsavev2_twinned_q_network_1_q_network_3_dense1_kernel_read_readvariableopJ
Fsavev2_twinned_q_network_1_q_network_3_dense1_bias_read_readvariableopL
Hsavev2_twinned_q_network_1_q_network_3_output_kernel_read_readvariableopJ
Fsavev2_twinned_q_network_1_q_network_3_output_bias_read_readvariableopL
Hsavev2_twinned_q_network_1_q_network_3_dense2_kernel_read_readvariableopJ
Fsavev2_twinned_q_network_1_q_network_3_dense2_bias_read_readvariableop
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
ShardedFilenameû
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¢
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices²
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Hsavev2_twinned_q_network_1_q_network_2_dense1_kernel_read_readvariableopFsavev2_twinned_q_network_1_q_network_2_dense1_bias_read_readvariableopHsavev2_twinned_q_network_1_q_network_2_output_kernel_read_readvariableopFsavev2_twinned_q_network_1_q_network_2_output_bias_read_readvariableopHsavev2_twinned_q_network_1_q_network_2_dense2_kernel_read_readvariableopFsavev2_twinned_q_network_1_q_network_2_dense2_bias_read_readvariableopHsavev2_twinned_q_network_1_q_network_3_dense1_kernel_read_readvariableopFsavev2_twinned_q_network_1_q_network_3_dense1_bias_read_readvariableopHsavev2_twinned_q_network_1_q_network_3_output_kernel_read_readvariableopFsavev2_twinned_q_network_1_q_network_3_output_bias_read_readvariableopHsavev2_twinned_q_network_1_q_network_3_dense2_kernel_read_readvariableopFsavev2_twinned_q_network_1_q_network_3_dense2_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
p: :	::	::
::	::	::
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	:!

_output_shapes	
::%	!

_output_shapes
:	: 


_output_shapes
::&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: 
´

ô
@__inference_dense2_layer_call_and_return_conditional_losses_3094

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

ô
@__inference_dense2_layer_call_and_return_conditional_losses_3405

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
°

ó
@__inference_dense1_layer_call_and_return_conditional_losses_3077

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°

ó
@__inference_dense1_layer_call_and_return_conditional_losses_3180

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


%__inference_output_layer_call_fn_3473

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallõ
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
GPU2 *0J 8 *I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_32132
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

»
"__inference_signature_wrapper_3365
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity

identity_1¢StatefulPartitionedCallì
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
GPU2 *0J 8 *(
f#R!
__inference__wrapped_model_30622
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
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1


%__inference_dense1_layer_call_fn_3374

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallö
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
GPU2 *0J 8 *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_30772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´

ô
@__inference_dense2_layer_call_and_return_conditional_losses_3197

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
Ã
Ë
2__inference_twinned_q_network_1_layer_call_fn_3332
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity

identity_1¢StatefulPartitionedCall
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
GPU2 *0J 8 *V
fQRO
M__inference_twinned_q_network_1_layer_call_and_return_conditional_losses_33002
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
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1


%__inference_output_layer_call_fn_3414

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallõ
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
GPU2 *0J 8 *I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_31102
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
d
ï
__inference__wrapped_model_3062
input_1X
Etwinned_q_network_1_q_network_2_dense1_matmul_readvariableop_resource:	U
Ftwinned_q_network_1_q_network_2_dense1_biasadd_readvariableop_resource:	Y
Etwinned_q_network_1_q_network_2_dense2_matmul_readvariableop_resource:
U
Ftwinned_q_network_1_q_network_2_dense2_biasadd_readvariableop_resource:	X
Etwinned_q_network_1_q_network_2_output_matmul_readvariableop_resource:	T
Ftwinned_q_network_1_q_network_2_output_biasadd_readvariableop_resource:X
Etwinned_q_network_1_q_network_3_dense1_matmul_readvariableop_resource:	U
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
:	*
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
:	*
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
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2~
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
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
°

ó
@__inference_dense1_layer_call_and_return_conditional_losses_3444

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
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
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ<
output_20
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:à¶

Q1
Q2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
j__call__
*k&call_and_return_all_conditional_losses
l_default_save_signature"±
_tf_keras_model{"name": "twinned_q_network_1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "TwinnedQNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 23]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "TwinnedQNetwork"}}
Ý
	hidden_units


dense1

dense2
out
trainable_variables
	variables
regularization_losses
	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_model{"name": "q_network_2", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "QNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 23]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "QNetwork"}}
Ý
hidden_units

dense1

dense2
out
trainable_variables
	variables
regularization_losses
	keras_api
o__call__
*p&call_and_return_all_conditional_losses"
_tf_keras_model{"name": "q_network_3", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "QNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 23]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "QNetwork"}}
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
%non_trainable_variables

&layers
trainable_variables
'layer_regularization_losses
	variables
regularization_losses
(metrics
)layer_metrics
j__call__
l_default_save_signature
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
,
qserving_default"
signature_map
 "
trackable_list_wrapper
È

kernel
bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
r__call__
*s&call_and_return_all_conditional_losses"£
_tf_keras_layer{"name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 23}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 23]}}
Ê

kernel
bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
t__call__
*u&call_and_return_all_conditional_losses"¥
_tf_keras_layer{"name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
Ì

kernel
bias
2trainable_variables
3	variables
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
6non_trainable_variables

7layers
trainable_variables
8layer_regularization_losses
	variables
regularization_losses
9metrics
:layer_metrics
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
Ì

kernel
 bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
x__call__
*y&call_and_return_all_conditional_losses"§
_tf_keras_layer{"name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 23}}, "shared_object_id": 15}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 23]}}
Î

#kernel
$bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
z__call__
*{&call_and_return_all_conditional_losses"©
_tf_keras_layer{"name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
Î

!kernel
"bias
Ctrainable_variables
D	variables
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
Gnon_trainable_variables

Hlayers
trainable_variables
Ilayer_regularization_losses
	variables
regularization_losses
Jmetrics
Klayer_metrics
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
@:>	2-twinned_q_network_1/q_network_2/dense1/kernel
::82+twinned_q_network_1/q_network_2/dense1/bias
@:>	2-twinned_q_network_1/q_network_2/output/kernel
9:72+twinned_q_network_1/q_network_2/output/bias
A:?
2-twinned_q_network_1/q_network_2/dense2/kernel
::82+twinned_q_network_1/q_network_2/dense2/bias
@:>	2-twinned_q_network_1/q_network_3/dense1/kernel
::82+twinned_q_network_1/q_network_3/dense1/bias
@:>	2-twinned_q_network_1/q_network_3/output/kernel
9:72+twinned_q_network_1/q_network_3/output/bias
A:?
2-twinned_q_network_1/q_network_3/dense2/kernel
::82+twinned_q_network_1/q_network_3/dense2/bias
 "
trackable_list_wrapper
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
Lnon_trainable_variables

Mlayers
*trainable_variables
Nlayer_regularization_losses
+	variables
,regularization_losses
Ometrics
Player_metrics
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
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
Qnon_trainable_variables

Rlayers
.trainable_variables
Slayer_regularization_losses
/	variables
0regularization_losses
Tmetrics
Ulayer_metrics
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
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
Vnon_trainable_variables

Wlayers
2trainable_variables
Xlayer_regularization_losses
3	variables
4regularization_losses
Ymetrics
Zlayer_metrics
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
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
[non_trainable_variables

\layers
;trainable_variables
]layer_regularization_losses
<	variables
=regularization_losses
^metrics
_layer_metrics
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
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
`non_trainable_variables

alayers
?trainable_variables
blayer_regularization_losses
@	variables
Aregularization_losses
cmetrics
dlayer_metrics
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
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
enon_trainable_variables

flayers
Ctrainable_variables
glayer_regularization_losses
D	variables
Eregularization_losses
hmetrics
ilayer_metrics
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
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
û2ø
2__inference_twinned_q_network_1_layer_call_fn_3332Á
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
input_1ÿÿÿÿÿÿÿÿÿ
2
M__inference_twinned_q_network_1_layer_call_and_return_conditional_losses_3300Á
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
input_1ÿÿÿÿÿÿÿÿÿ
Ý2Ú
__inference__wrapped_model_3062¶
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
input_1ÿÿÿÿÿÿÿÿÿ
ó2ð
*__inference_q_network_2_layer_call_fn_3135Á
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
input_1ÿÿÿÿÿÿÿÿÿ
2
E__inference_q_network_2_layer_call_and_return_conditional_losses_3117Á
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
input_1ÿÿÿÿÿÿÿÿÿ
ó2ð
*__inference_q_network_3_layer_call_fn_3238Á
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
input_1ÿÿÿÿÿÿÿÿÿ
2
E__inference_q_network_3_layer_call_and_return_conditional_losses_3220Á
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
input_1ÿÿÿÿÿÿÿÿÿ
ÉBÆ
"__inference_signature_wrapper_3365input_1"
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
Ï2Ì
%__inference_dense1_layer_call_fn_3374¢
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
ê2ç
@__inference_dense1_layer_call_and_return_conditional_losses_3385¢
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
Ï2Ì
%__inference_dense2_layer_call_fn_3394¢
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
ê2ç
@__inference_dense2_layer_call_and_return_conditional_losses_3405¢
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
Ï2Ì
%__inference_output_layer_call_fn_3414¢
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
ê2ç
@__inference_output_layer_call_and_return_conditional_losses_3424¢
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
Ï2Ì
%__inference_dense1_layer_call_fn_3433¢
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
ê2ç
@__inference_dense1_layer_call_and_return_conditional_losses_3444¢
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
Ï2Ì
%__inference_dense2_layer_call_fn_3453¢
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
ê2ç
@__inference_dense2_layer_call_and_return_conditional_losses_3464¢
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
Ï2Ì
%__inference_output_layer_call_fn_3473¢
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
ê2ç
@__inference_output_layer_call_and_return_conditional_losses_3483¢
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
 É
__inference__wrapped_model_3062¥ #$!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "cª`
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿ¡
@__inference_dense1_layer_call_and_return_conditional_losses_3385]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¡
@__inference_dense1_layer_call_and_return_conditional_losses_3444] /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 y
%__inference_dense1_layer_call_fn_3374P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿy
%__inference_dense1_layer_call_fn_3433P /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¢
@__inference_dense2_layer_call_and_return_conditional_losses_3405^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¢
@__inference_dense2_layer_call_and_return_conditional_losses_3464^#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 z
%__inference_dense2_layer_call_fn_3394Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿz
%__inference_dense2_layer_call_fn_3453Q#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¡
@__inference_output_layer_call_and_return_conditional_losses_3424]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¡
@__inference_output_layer_call_and_return_conditional_losses_3483]!"0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
%__inference_output_layer_call_fn_3414P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿy
%__inference_output_layer_call_fn_3473P!"0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
E__inference_q_network_2_layer_call_and_return_conditional_losses_3117a0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_q_network_2_layer_call_fn_3135T0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
E__inference_q_network_3_layer_call_and_return_conditional_losses_3220a #$!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_q_network_3_layer_call_fn_3238T #$!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ×
"__inference_signature_wrapper_3365° #$!";¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"cª`
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿß
M__inference_twinned_q_network_1_layer_call_and_return_conditional_losses_3300 #$!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 µ
2__inference_twinned_q_network_1_layer_call_fn_3332 #$!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "=¢:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ