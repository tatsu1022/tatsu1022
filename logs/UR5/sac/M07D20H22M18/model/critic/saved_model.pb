??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
)twinned_q_network/q_network/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	%?*:
shared_name+)twinned_q_network/q_network/dense1/kernel
?
=twinned_q_network/q_network/dense1/kernel/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network/dense1/kernel*
_output_shapes
:	%?*
dtype0
?
'twinned_q_network/q_network/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'twinned_q_network/q_network/dense1/bias
?
;twinned_q_network/q_network/dense1/bias/Read/ReadVariableOpReadVariableOp'twinned_q_network/q_network/dense1/bias*
_output_shapes	
:?*
dtype0
?
)twinned_q_network/q_network/output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*:
shared_name+)twinned_q_network/q_network/output/kernel
?
=twinned_q_network/q_network/output/kernel/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network/output/kernel*
_output_shapes
:	?*
dtype0
?
'twinned_q_network/q_network/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'twinned_q_network/q_network/output/bias
?
;twinned_q_network/q_network/output/bias/Read/ReadVariableOpReadVariableOp'twinned_q_network/q_network/output/bias*
_output_shapes
:*
dtype0
?
)twinned_q_network/q_network/dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*:
shared_name+)twinned_q_network/q_network/dense2/kernel
?
=twinned_q_network/q_network/dense2/kernel/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network/dense2/kernel* 
_output_shapes
:
??*
dtype0
?
'twinned_q_network/q_network/dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'twinned_q_network/q_network/dense2/bias
?
;twinned_q_network/q_network/dense2/bias/Read/ReadVariableOpReadVariableOp'twinned_q_network/q_network/dense2/bias*
_output_shapes	
:?*
dtype0
?
+twinned_q_network/q_network_1/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	%?*<
shared_name-+twinned_q_network/q_network_1/dense1/kernel
?
?twinned_q_network/q_network_1/dense1/kernel/Read/ReadVariableOpReadVariableOp+twinned_q_network/q_network_1/dense1/kernel*
_output_shapes
:	%?*
dtype0
?
)twinned_q_network/q_network_1/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)twinned_q_network/q_network_1/dense1/bias
?
=twinned_q_network/q_network_1/dense1/bias/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network_1/dense1/bias*
_output_shapes	
:?*
dtype0
?
+twinned_q_network/q_network_1/output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*<
shared_name-+twinned_q_network/q_network_1/output/kernel
?
?twinned_q_network/q_network_1/output/kernel/Read/ReadVariableOpReadVariableOp+twinned_q_network/q_network_1/output/kernel*
_output_shapes
:	?*
dtype0
?
)twinned_q_network/q_network_1/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)twinned_q_network/q_network_1/output/bias
?
=twinned_q_network/q_network_1/output/bias/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network_1/output/bias*
_output_shapes
:*
dtype0
?
+twinned_q_network/q_network_1/dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*<
shared_name-+twinned_q_network/q_network_1/dense2/kernel
?
?twinned_q_network/q_network_1/dense2/kernel/Read/ReadVariableOpReadVariableOp+twinned_q_network/q_network_1/dense2/kernel* 
_output_shapes
:
??*
dtype0
?
)twinned_q_network/q_network_1/dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)twinned_q_network/q_network_1/dense2/bias
?
=twinned_q_network/q_network_1/dense2/bias/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network_1/dense2/bias*
_output_shapes	
:?*
dtype0

NoOpNoOp
?%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?$
value?$B?$ B?$
?
Q1
Q2
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
?
	hidden_units


dense1

dense2
out
trainable_variables
regularization_losses
	variables
	keras_api
?
hidden_units

dense1

dense2
out
trainable_variables
regularization_losses
	variables
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
 
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
?
trainable_variables
regularization_losses
%non_trainable_variables

&layers
	variables
'layer_regularization_losses
(metrics
)layer_metrics
 
 
h

kernel
bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
h

kernel
bias
.trainable_variables
/regularization_losses
0	variables
1	keras_api
h

kernel
bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
?
trainable_variables
regularization_losses
6non_trainable_variables

7layers
	variables
8layer_regularization_losses
9metrics
:layer_metrics
 
h

kernel
 bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
h

#kernel
$bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
h

!kernel
"bias
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
*
0
 1
!2
"3
#4
$5
 
*
0
 1
#2
$3
!4
"5
?
trainable_variables
regularization_losses
Gnon_trainable_variables

Hlayers
	variables
Ilayer_regularization_losses
Jmetrics
Klayer_metrics
om
VARIABLE_VALUE)twinned_q_network/q_network/dense1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'twinned_q_network/q_network/dense1/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE)twinned_q_network/q_network/output/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'twinned_q_network/q_network/output/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE)twinned_q_network/q_network/dense2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'twinned_q_network/q_network/dense2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE+twinned_q_network/q_network_1/dense1/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE)twinned_q_network/q_network_1/dense1/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE+twinned_q_network/q_network_1/output/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE)twinned_q_network/q_network_1/output/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE+twinned_q_network/q_network_1/dense2/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE)twinned_q_network/q_network_1/dense2/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
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
 

0
1
?
*trainable_variables
+regularization_losses
Lnon_trainable_variables

Mlayers
,	variables
Nlayer_regularization_losses
Ometrics
Player_metrics

0
1
 

0
1
?
.trainable_variables
/regularization_losses
Qnon_trainable_variables

Rlayers
0	variables
Slayer_regularization_losses
Tmetrics
Ulayer_metrics

0
1
 

0
1
?
2trainable_variables
3regularization_losses
Vnon_trainable_variables

Wlayers
4	variables
Xlayer_regularization_losses
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
 

0
 1
?
;trainable_variables
<regularization_losses
[non_trainable_variables

\layers
=	variables
]layer_regularization_losses
^metrics
_layer_metrics

#0
$1
 

#0
$1
?
?trainable_variables
@regularization_losses
`non_trainable_variables

alayers
A	variables
blayer_regularization_losses
cmetrics
dlayer_metrics

!0
"1
 

!0
"1
?
Ctrainable_variables
Dregularization_losses
enon_trainable_variables

flayers
E	variables
glayer_regularization_losses
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
:?????????%*
dtype0*
shape:?????????%
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1)twinned_q_network/q_network/dense1/kernel'twinned_q_network/q_network/dense1/bias)twinned_q_network/q_network/dense2/kernel'twinned_q_network/q_network/dense2/bias)twinned_q_network/q_network/output/kernel'twinned_q_network/q_network/output/bias+twinned_q_network/q_network_1/dense1/kernel)twinned_q_network/q_network_1/dense1/bias+twinned_q_network/q_network_1/dense2/kernel)twinned_q_network/q_network_1/dense2/bias+twinned_q_network/q_network_1/output/kernel)twinned_q_network/q_network_1/output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? */
f*R(
&__inference_signature_wrapper_26803475
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename=twinned_q_network/q_network/dense1/kernel/Read/ReadVariableOp;twinned_q_network/q_network/dense1/bias/Read/ReadVariableOp=twinned_q_network/q_network/output/kernel/Read/ReadVariableOp;twinned_q_network/q_network/output/bias/Read/ReadVariableOp=twinned_q_network/q_network/dense2/kernel/Read/ReadVariableOp;twinned_q_network/q_network/dense2/bias/Read/ReadVariableOp?twinned_q_network/q_network_1/dense1/kernel/Read/ReadVariableOp=twinned_q_network/q_network_1/dense1/bias/Read/ReadVariableOp?twinned_q_network/q_network_1/output/kernel/Read/ReadVariableOp=twinned_q_network/q_network_1/output/bias/Read/ReadVariableOp?twinned_q_network/q_network_1/dense2/kernel/Read/ReadVariableOp=twinned_q_network/q_network_1/dense2/bias/Read/ReadVariableOpConst*
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
GPU2 *0J 8? **
f%R#
!__inference__traced_save_26803653
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename)twinned_q_network/q_network/dense1/kernel'twinned_q_network/q_network/dense1/bias)twinned_q_network/q_network/output/kernel'twinned_q_network/q_network/output/bias)twinned_q_network/q_network/dense2/kernel'twinned_q_network/q_network/dense2/bias+twinned_q_network/q_network_1/dense1/kernel)twinned_q_network/q_network_1/dense1/bias+twinned_q_network/q_network_1/output/kernel)twinned_q_network/q_network_1/output/bias+twinned_q_network/q_network_1/dense2/kernel)twinned_q_network/q_network_1/dense2/bias*
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
GPU2 *0J 8? *-
f(R&
$__inference__traced_restore_26803699??
?
?
I__inference_q_network_1_layer_call_and_return_conditional_losses_26803330
input_1"
dense1_26803291:	%?
dense1_26803293:	?#
dense2_26803308:
??
dense2_26803310:	?"
output_26803324:	?
output_26803326:
identity??dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?output/StatefulPartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_26803291dense1_26803293*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_268032902 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_26803308dense2_26803310*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense2_layer_call_and_return_conditional_losses_268033072 
dense2/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_26803324output_26803326*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_268033232 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????%: : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
'
_output_shapes
:?????????%
!
_user_specified_name	input_1
?
?
4__inference_twinned_q_network_layer_call_fn_26803442
input_1
unknown:	%?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?
	unknown_4:
	unknown_5:	%?
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_twinned_q_network_layer_call_and_return_conditional_losses_268034102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????%: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????%
!
_user_specified_name	input_1
?
?
G__inference_q_network_layer_call_and_return_conditional_losses_26803227
input_1"
dense1_26803188:	%?
dense1_26803190:	?#
dense2_26803205:
??
dense2_26803207:	?"
output_26803221:	?
output_26803223:
identity??dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?output/StatefulPartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_26803188dense1_26803190*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_268031872 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_26803205dense2_26803207*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense2_layer_call_and_return_conditional_losses_268032042 
dense2/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_26803221output_26803223*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_268032202 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????%: : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
'
_output_shapes
:?????????%
!
_user_specified_name	input_1
?
?
.__inference_q_network_1_layer_call_fn_26803348
input_1
unknown:	%?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_q_network_1_layer_call_and_return_conditional_losses_268033302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????%: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????%
!
_user_specified_name	input_1
?

?
D__inference_dense1_layer_call_and_return_conditional_losses_26803486

inputs1
matmul_readvariableop_resource:	%?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	%?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????%: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????%
 
_user_specified_nameinputs
?
?
)__inference_output_layer_call_fn_26803593

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_268033232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_26803475
input_1
unknown:	%?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?
	unknown_4:
	unknown_5:	%?
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *,
f'R%
#__inference__wrapped_model_268031722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????%: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????%
!
_user_specified_name	input_1
?
?
,__inference_q_network_layer_call_fn_26803245
input_1
unknown:	%?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_q_network_layer_call_and_return_conditional_losses_268032272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????%: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????%
!
_user_specified_name	input_1
?	
?
D__inference_output_layer_call_and_return_conditional_losses_26803220

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_output_layer_call_and_return_conditional_losses_26803323

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense1_layer_call_and_return_conditional_losses_26803290

inputs1
matmul_readvariableop_resource:	%?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	%?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????%: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????%
 
_user_specified_nameinputs
?)
?
!__inference__traced_save_26803653
file_prefixH
Dsavev2_twinned_q_network_q_network_dense1_kernel_read_readvariableopF
Bsavev2_twinned_q_network_q_network_dense1_bias_read_readvariableopH
Dsavev2_twinned_q_network_q_network_output_kernel_read_readvariableopF
Bsavev2_twinned_q_network_q_network_output_bias_read_readvariableopH
Dsavev2_twinned_q_network_q_network_dense2_kernel_read_readvariableopF
Bsavev2_twinned_q_network_q_network_dense2_bias_read_readvariableopJ
Fsavev2_twinned_q_network_q_network_1_dense1_kernel_read_readvariableopH
Dsavev2_twinned_q_network_q_network_1_dense1_bias_read_readvariableopJ
Fsavev2_twinned_q_network_q_network_1_output_kernel_read_readvariableopH
Dsavev2_twinned_q_network_q_network_1_output_bias_read_readvariableopJ
Fsavev2_twinned_q_network_q_network_1_dense2_kernel_read_readvariableopH
Dsavev2_twinned_q_network_q_network_1_dense2_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Dsavev2_twinned_q_network_q_network_dense1_kernel_read_readvariableopBsavev2_twinned_q_network_q_network_dense1_bias_read_readvariableopDsavev2_twinned_q_network_q_network_output_kernel_read_readvariableopBsavev2_twinned_q_network_q_network_output_bias_read_readvariableopDsavev2_twinned_q_network_q_network_dense2_kernel_read_readvariableopBsavev2_twinned_q_network_q_network_dense2_bias_read_readvariableopFsavev2_twinned_q_network_q_network_1_dense1_kernel_read_readvariableopDsavev2_twinned_q_network_q_network_1_dense1_bias_read_readvariableopFsavev2_twinned_q_network_q_network_1_output_kernel_read_readvariableopDsavev2_twinned_q_network_q_network_1_output_bias_read_readvariableopFsavev2_twinned_q_network_q_network_1_dense2_kernel_read_readvariableopDsavev2_twinned_q_network_q_network_1_dense2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapesr
p: :	%?:?:	?::
??:?:	%?:?:	?::
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	%?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	%?:!

_output_shapes	
:?:%	!

_output_shapes
:	?: 


_output_shapes
::&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: 
?

?
D__inference_dense2_layer_call_and_return_conditional_losses_26803506

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense1_layer_call_and_return_conditional_losses_26803545

inputs1
matmul_readvariableop_resource:	%?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	%?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????%: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????%
 
_user_specified_nameinputs
?`
?
#__inference__wrapped_model_26803172
input_1T
Atwinned_q_network_q_network_dense1_matmul_readvariableop_resource:	%?Q
Btwinned_q_network_q_network_dense1_biasadd_readvariableop_resource:	?U
Atwinned_q_network_q_network_dense2_matmul_readvariableop_resource:
??Q
Btwinned_q_network_q_network_dense2_biasadd_readvariableop_resource:	?T
Atwinned_q_network_q_network_output_matmul_readvariableop_resource:	?P
Btwinned_q_network_q_network_output_biasadd_readvariableop_resource:V
Ctwinned_q_network_q_network_1_dense1_matmul_readvariableop_resource:	%?S
Dtwinned_q_network_q_network_1_dense1_biasadd_readvariableop_resource:	?W
Ctwinned_q_network_q_network_1_dense2_matmul_readvariableop_resource:
??S
Dtwinned_q_network_q_network_1_dense2_biasadd_readvariableop_resource:	?V
Ctwinned_q_network_q_network_1_output_matmul_readvariableop_resource:	?R
Dtwinned_q_network_q_network_1_output_biasadd_readvariableop_resource:
identity

identity_1??9twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOp?8twinned_q_network/q_network/dense1/MatMul/ReadVariableOp?9twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOp?8twinned_q_network/q_network/dense2/MatMul/ReadVariableOp?9twinned_q_network/q_network/output/BiasAdd/ReadVariableOp?8twinned_q_network/q_network/output/MatMul/ReadVariableOp?;twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOp?:twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOp?;twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOp?:twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOp?;twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOp?:twinned_q_network/q_network_1/output/MatMul/ReadVariableOp?
8twinned_q_network/q_network/dense1/MatMul/ReadVariableOpReadVariableOpAtwinned_q_network_q_network_dense1_matmul_readvariableop_resource*
_output_shapes
:	%?*
dtype02:
8twinned_q_network/q_network/dense1/MatMul/ReadVariableOp?
)twinned_q_network/q_network/dense1/MatMulMatMulinput_1@twinned_q_network/q_network/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)twinned_q_network/q_network/dense1/MatMul?
9twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOpReadVariableOpBtwinned_q_network_q_network_dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOp?
*twinned_q_network/q_network/dense1/BiasAddBiasAdd3twinned_q_network/q_network/dense1/MatMul:product:0Atwinned_q_network/q_network/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*twinned_q_network/q_network/dense1/BiasAdd?
'twinned_q_network/q_network/dense1/ReluRelu3twinned_q_network/q_network/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2)
'twinned_q_network/q_network/dense1/Relu?
8twinned_q_network/q_network/dense2/MatMul/ReadVariableOpReadVariableOpAtwinned_q_network_q_network_dense2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02:
8twinned_q_network/q_network/dense2/MatMul/ReadVariableOp?
)twinned_q_network/q_network/dense2/MatMulMatMul5twinned_q_network/q_network/dense1/Relu:activations:0@twinned_q_network/q_network/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)twinned_q_network/q_network/dense2/MatMul?
9twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOpReadVariableOpBtwinned_q_network_q_network_dense2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOp?
*twinned_q_network/q_network/dense2/BiasAddBiasAdd3twinned_q_network/q_network/dense2/MatMul:product:0Atwinned_q_network/q_network/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*twinned_q_network/q_network/dense2/BiasAdd?
'twinned_q_network/q_network/dense2/ReluRelu3twinned_q_network/q_network/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2)
'twinned_q_network/q_network/dense2/Relu?
8twinned_q_network/q_network/output/MatMul/ReadVariableOpReadVariableOpAtwinned_q_network_q_network_output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02:
8twinned_q_network/q_network/output/MatMul/ReadVariableOp?
)twinned_q_network/q_network/output/MatMulMatMul5twinned_q_network/q_network/dense2/Relu:activations:0@twinned_q_network/q_network/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)twinned_q_network/q_network/output/MatMul?
9twinned_q_network/q_network/output/BiasAdd/ReadVariableOpReadVariableOpBtwinned_q_network_q_network_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9twinned_q_network/q_network/output/BiasAdd/ReadVariableOp?
*twinned_q_network/q_network/output/BiasAddBiasAdd3twinned_q_network/q_network/output/MatMul:product:0Atwinned_q_network/q_network/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2,
*twinned_q_network/q_network/output/BiasAdd?
:twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOpReadVariableOpCtwinned_q_network_q_network_1_dense1_matmul_readvariableop_resource*
_output_shapes
:	%?*
dtype02<
:twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOp?
+twinned_q_network/q_network_1/dense1/MatMulMatMulinput_1Btwinned_q_network/q_network_1/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+twinned_q_network/q_network_1/dense1/MatMul?
;twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOpReadVariableOpDtwinned_q_network_q_network_1_dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOp?
,twinned_q_network/q_network_1/dense1/BiasAddBiasAdd5twinned_q_network/q_network_1/dense1/MatMul:product:0Ctwinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,twinned_q_network/q_network_1/dense1/BiasAdd?
)twinned_q_network/q_network_1/dense1/ReluRelu5twinned_q_network/q_network_1/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2+
)twinned_q_network/q_network_1/dense1/Relu?
:twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOpReadVariableOpCtwinned_q_network_q_network_1_dense2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02<
:twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOp?
+twinned_q_network/q_network_1/dense2/MatMulMatMul7twinned_q_network/q_network_1/dense1/Relu:activations:0Btwinned_q_network/q_network_1/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+twinned_q_network/q_network_1/dense2/MatMul?
;twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOpReadVariableOpDtwinned_q_network_q_network_1_dense2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOp?
,twinned_q_network/q_network_1/dense2/BiasAddBiasAdd5twinned_q_network/q_network_1/dense2/MatMul:product:0Ctwinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,twinned_q_network/q_network_1/dense2/BiasAdd?
)twinned_q_network/q_network_1/dense2/ReluRelu5twinned_q_network/q_network_1/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2+
)twinned_q_network/q_network_1/dense2/Relu?
:twinned_q_network/q_network_1/output/MatMul/ReadVariableOpReadVariableOpCtwinned_q_network_q_network_1_output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02<
:twinned_q_network/q_network_1/output/MatMul/ReadVariableOp?
+twinned_q_network/q_network_1/output/MatMulMatMul7twinned_q_network/q_network_1/dense2/Relu:activations:0Btwinned_q_network/q_network_1/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+twinned_q_network/q_network_1/output/MatMul?
;twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOpReadVariableOpDtwinned_q_network_q_network_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOp?
,twinned_q_network/q_network_1/output/BiasAddBiasAdd5twinned_q_network/q_network_1/output/MatMul:product:0Ctwinned_q_network/q_network_1/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,twinned_q_network/q_network_1/output/BiasAdd?
IdentityIdentity3twinned_q_network/q_network/output/BiasAdd:output:0:^twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/dense1/MatMul/ReadVariableOp:^twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/dense2/MatMul/ReadVariableOp:^twinned_q_network/q_network/output/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/output/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity5twinned_q_network/q_network_1/output/BiasAdd:output:0:^twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/dense1/MatMul/ReadVariableOp:^twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/dense2/MatMul/ReadVariableOp:^twinned_q_network/q_network/output/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/output/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????%: : : : : : : : : : : : 2v
9twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOp9twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOp2t
8twinned_q_network/q_network/dense1/MatMul/ReadVariableOp8twinned_q_network/q_network/dense1/MatMul/ReadVariableOp2v
9twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOp9twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOp2t
8twinned_q_network/q_network/dense2/MatMul/ReadVariableOp8twinned_q_network/q_network/dense2/MatMul/ReadVariableOp2v
9twinned_q_network/q_network/output/BiasAdd/ReadVariableOp9twinned_q_network/q_network/output/BiasAdd/ReadVariableOp2t
8twinned_q_network/q_network/output/MatMul/ReadVariableOp8twinned_q_network/q_network/output/MatMul/ReadVariableOp2z
;twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOp;twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOp2x
:twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOp:twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOp2z
;twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOp;twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOp2x
:twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOp:twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOp2z
;twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOp;twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOp2x
:twinned_q_network/q_network_1/output/MatMul/ReadVariableOp:twinned_q_network/q_network_1/output/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????%
!
_user_specified_name	input_1
?:
?	
$__inference__traced_restore_26803699
file_prefixM
:assignvariableop_twinned_q_network_q_network_dense1_kernel:	%?I
:assignvariableop_1_twinned_q_network_q_network_dense1_bias:	?O
<assignvariableop_2_twinned_q_network_q_network_output_kernel:	?H
:assignvariableop_3_twinned_q_network_q_network_output_bias:P
<assignvariableop_4_twinned_q_network_q_network_dense2_kernel:
??I
:assignvariableop_5_twinned_q_network_q_network_dense2_bias:	?Q
>assignvariableop_6_twinned_q_network_q_network_1_dense1_kernel:	%?K
<assignvariableop_7_twinned_q_network_q_network_1_dense1_bias:	?Q
>assignvariableop_8_twinned_q_network_q_network_1_output_kernel:	?J
<assignvariableop_9_twinned_q_network_q_network_1_output_bias:S
?assignvariableop_10_twinned_q_network_q_network_1_dense2_kernel:
??L
=assignvariableop_11_twinned_q_network_q_network_1_dense2_bias:	?
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
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

Identity?
AssignVariableOpAssignVariableOp:assignvariableop_twinned_q_network_q_network_dense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp:assignvariableop_1_twinned_q_network_q_network_dense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp<assignvariableop_2_twinned_q_network_q_network_output_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp:assignvariableop_3_twinned_q_network_q_network_output_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp<assignvariableop_4_twinned_q_network_q_network_dense2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp:assignvariableop_5_twinned_q_network_q_network_dense2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp>assignvariableop_6_twinned_q_network_q_network_1_dense1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp<assignvariableop_7_twinned_q_network_q_network_1_dense1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp>assignvariableop_8_twinned_q_network_q_network_1_output_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp<assignvariableop_9_twinned_q_network_q_network_1_output_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp?assignvariableop_10_twinned_q_network_q_network_1_dense2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp=assignvariableop_11_twinned_q_network_q_network_1_dense2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12?
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
?

?
D__inference_dense2_layer_call_and_return_conditional_losses_26803307

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
O__inference_twinned_q_network_layer_call_and_return_conditional_losses_26803410
input_1%
q_network_26803382:	%?!
q_network_26803384:	?&
q_network_26803386:
??!
q_network_26803388:	?%
q_network_26803390:	? 
q_network_26803392:'
q_network_1_26803395:	%?#
q_network_1_26803397:	?(
q_network_1_26803399:
??#
q_network_1_26803401:	?'
q_network_1_26803403:	?"
q_network_1_26803405:
identity

identity_1??!q_network/StatefulPartitionedCall?#q_network_1/StatefulPartitionedCall?
!q_network/StatefulPartitionedCallStatefulPartitionedCallinput_1q_network_26803382q_network_26803384q_network_26803386q_network_26803388q_network_26803390q_network_26803392*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_q_network_layer_call_and_return_conditional_losses_268032272#
!q_network/StatefulPartitionedCall?
#q_network_1/StatefulPartitionedCallStatefulPartitionedCallinput_1q_network_1_26803395q_network_1_26803397q_network_1_26803399q_network_1_26803401q_network_1_26803403q_network_1_26803405*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_q_network_1_layer_call_and_return_conditional_losses_268033302%
#q_network_1/StatefulPartitionedCall?
IdentityIdentity*q_network/StatefulPartitionedCall:output:0"^q_network/StatefulPartitionedCall$^q_network_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity,q_network_1/StatefulPartitionedCall:output:0"^q_network/StatefulPartitionedCall$^q_network_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????%: : : : : : : : : : : : 2F
!q_network/StatefulPartitionedCall!q_network/StatefulPartitionedCall2J
#q_network_1/StatefulPartitionedCall#q_network_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????%
!
_user_specified_name	input_1
?
?
)__inference_dense1_layer_call_fn_26803554

inputs
unknown:	%?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_268032902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????%: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????%
 
_user_specified_nameinputs
?
?
)__inference_dense2_layer_call_fn_26803515

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense2_layer_call_and_return_conditional_losses_268032042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_dense2_layer_call_fn_26803574

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense2_layer_call_and_return_conditional_losses_268033072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense1_layer_call_and_return_conditional_losses_26803187

inputs1
matmul_readvariableop_resource:	%?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	%?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????%: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????%
 
_user_specified_nameinputs
?	
?
D__inference_output_layer_call_and_return_conditional_losses_26803525

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_output_layer_call_and_return_conditional_losses_26803584

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_dense1_layer_call_fn_26803495

inputs
unknown:	%?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_268031872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????%: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????%
 
_user_specified_nameinputs
?
?
)__inference_output_layer_call_fn_26803534

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_268032202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense2_layer_call_and_return_conditional_losses_26803204

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense2_layer_call_and_return_conditional_losses_26803565

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????%<
output_10
StatefulPartitionedCall:0?????????<
output_20
StatefulPartitionedCall:1?????????tensorflow/serving/predict:ȷ
?
Q1
Q2
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
*j&call_and_return_all_conditional_losses
k__call__
l_default_save_signature"?
_tf_keras_model?{"name": "twinned_q_network", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "TwinnedQNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 37]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "TwinnedQNetwork"}}
?
	hidden_units


dense1

dense2
out
trainable_variables
regularization_losses
	variables
	keras_api
*m&call_and_return_all_conditional_losses
n__call__"?
_tf_keras_model?{"name": "q_network", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "QNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 37]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "QNetwork"}}
?
hidden_units

dense1

dense2
out
trainable_variables
regularization_losses
	variables
	keras_api
*o&call_and_return_all_conditional_losses
p__call__"?
_tf_keras_model?{"name": "q_network_1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "QNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 37]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "QNetwork"}}
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
 "
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
?
trainable_variables
regularization_losses
%non_trainable_variables

&layers
	variables
'layer_regularization_losses
(metrics
)layer_metrics
k__call__
l_default_save_signature
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
,
qserving_default"
signature_map
 "
trackable_list_wrapper
?

kernel
bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
*r&call_and_return_all_conditional_losses
s__call__"?
_tf_keras_layer?{"name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 37}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 37]}}
?

kernel
bias
.trainable_variables
/regularization_losses
0	variables
1	keras_api
*t&call_and_return_all_conditional_losses
u__call__"?
_tf_keras_layer?{"name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
?

kernel
bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
*v&call_and_return_all_conditional_losses
w__call__"?
_tf_keras_layer?{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float64", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?
trainable_variables
regularization_losses
6non_trainable_variables

7layers
	variables
8layer_regularization_losses
9metrics
:layer_metrics
n__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
?

kernel
 bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
*x&call_and_return_all_conditional_losses
y__call__"?
_tf_keras_layer?{"name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 37}}, "shared_object_id": 15}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 37]}}
?

#kernel
$bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
*z&call_and_return_all_conditional_losses
{__call__"?
_tf_keras_layer?{"name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
?

!kernel
"bias
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
*|&call_and_return_all_conditional_losses
}__call__"?
_tf_keras_layer?{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float64", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
J
0
 1
!2
"3
#4
$5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
 1
#2
$3
!4
"5"
trackable_list_wrapper
?
trainable_variables
regularization_losses
Gnon_trainable_variables

Hlayers
	variables
Ilayer_regularization_losses
Jmetrics
Klayer_metrics
p__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
<::	%?2)twinned_q_network/q_network/dense1/kernel
6:4?2'twinned_q_network/q_network/dense1/bias
<::	?2)twinned_q_network/q_network/output/kernel
5:32'twinned_q_network/q_network/output/bias
=:;
??2)twinned_q_network/q_network/dense2/kernel
6:4?2'twinned_q_network/q_network/dense2/bias
>:<	%?2+twinned_q_network/q_network_1/dense1/kernel
8:6?2)twinned_q_network/q_network_1/dense1/bias
>:<	?2+twinned_q_network/q_network_1/output/kernel
7:52)twinned_q_network/q_network_1/output/bias
?:=
??2+twinned_q_network/q_network_1/dense2/kernel
8:6?2)twinned_q_network/q_network_1/dense2/bias
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
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
*trainable_variables
+regularization_losses
Lnon_trainable_variables

Mlayers
,	variables
Nlayer_regularization_losses
Ometrics
Player_metrics
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
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
?
.trainable_variables
/regularization_losses
Qnon_trainable_variables

Rlayers
0	variables
Slayer_regularization_losses
Tmetrics
Ulayer_metrics
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
2trainable_variables
3regularization_losses
Vnon_trainable_variables

Wlayers
4	variables
Xlayer_regularization_losses
Ymetrics
Zlayer_metrics
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
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
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
;trainable_variables
<regularization_losses
[non_trainable_variables

\layers
=	variables
]layer_regularization_losses
^metrics
_layer_metrics
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
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
?
?trainable_variables
@regularization_losses
`non_trainable_variables

alayers
A	variables
blayer_regularization_losses
cmetrics
dlayer_metrics
{__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?
Ctrainable_variables
Dregularization_losses
enon_trainable_variables

flayers
E	variables
glayer_regularization_losses
hmetrics
ilayer_metrics
}__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
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
?2?
O__inference_twinned_q_network_layer_call_and_return_conditional_losses_26803410?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????%
?2?
4__inference_twinned_q_network_layer_call_fn_26803442?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????%
?2?
#__inference__wrapped_model_26803172?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????%
?2?
G__inference_q_network_layer_call_and_return_conditional_losses_26803227?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????%
?2?
,__inference_q_network_layer_call_fn_26803245?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????%
?2?
I__inference_q_network_1_layer_call_and_return_conditional_losses_26803330?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????%
?2?
.__inference_q_network_1_layer_call_fn_26803348?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????%
?B?
&__inference_signature_wrapper_26803475input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense1_layer_call_and_return_conditional_losses_26803486?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense1_layer_call_fn_26803495?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense2_layer_call_and_return_conditional_losses_26803506?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense2_layer_call_fn_26803515?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_output_layer_call_and_return_conditional_losses_26803525?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_output_layer_call_fn_26803534?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense1_layer_call_and_return_conditional_losses_26803545?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense1_layer_call_fn_26803554?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense2_layer_call_and_return_conditional_losses_26803565?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense2_layer_call_fn_26803574?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_output_layer_call_and_return_conditional_losses_26803584?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_output_layer_call_fn_26803593?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_26803172? #$!"0?-
&?#
!?
input_1?????????%
? "c?`
.
output_1"?
output_1?????????
.
output_2"?
output_2??????????
D__inference_dense1_layer_call_and_return_conditional_losses_26803486]/?,
%?"
 ?
inputs?????????%
? "&?#
?
0??????????
? ?
D__inference_dense1_layer_call_and_return_conditional_losses_26803545] /?,
%?"
 ?
inputs?????????%
? "&?#
?
0??????????
? }
)__inference_dense1_layer_call_fn_26803495P/?,
%?"
 ?
inputs?????????%
? "???????????}
)__inference_dense1_layer_call_fn_26803554P /?,
%?"
 ?
inputs?????????%
? "????????????
D__inference_dense2_layer_call_and_return_conditional_losses_26803506^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
D__inference_dense2_layer_call_and_return_conditional_losses_26803565^#$0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense2_layer_call_fn_26803515Q0?-
&?#
!?
inputs??????????
? "???????????~
)__inference_dense2_layer_call_fn_26803574Q#$0?-
&?#
!?
inputs??????????
? "????????????
D__inference_output_layer_call_and_return_conditional_losses_26803525]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
D__inference_output_layer_call_and_return_conditional_losses_26803584]!"0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_output_layer_call_fn_26803534P0?-
&?#
!?
inputs??????????
? "??????????}
)__inference_output_layer_call_fn_26803593P!"0?-
&?#
!?
inputs??????????
? "???????????
I__inference_q_network_1_layer_call_and_return_conditional_losses_26803330a #$!"0?-
&?#
!?
input_1?????????%
? "%?"
?
0?????????
? ?
.__inference_q_network_1_layer_call_fn_26803348T #$!"0?-
&?#
!?
input_1?????????%
? "???????????
G__inference_q_network_layer_call_and_return_conditional_losses_26803227a0?-
&?#
!?
input_1?????????%
? "%?"
?
0?????????
? ?
,__inference_q_network_layer_call_fn_26803245T0?-
&?#
!?
input_1?????????%
? "???????????
&__inference_signature_wrapper_26803475? #$!";?8
? 
1?.
,
input_1!?
input_1?????????%"c?`
.
output_1"?
output_1?????????
.
output_2"?
output_2??????????
O__inference_twinned_q_network_layer_call_and_return_conditional_losses_26803410? #$!"0?-
&?#
!?
input_1?????????%
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
4__inference_twinned_q_network_layer_call_fn_26803442 #$!"0?-
&?#
!?
input_1?????????%
? "=?:
?
0?????????
?
1?????????