©î
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
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718¨§
¯
)twinned_q_network/q_network/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	#*:
shared_name+)twinned_q_network/q_network/dense1/kernel
¨
=twinned_q_network/q_network/dense1/kernel/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network/dense1/kernel*
_output_shapes
:	#*
dtype0
§
'twinned_q_network/q_network/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'twinned_q_network/q_network/dense1/bias
 
;twinned_q_network/q_network/dense1/bias/Read/ReadVariableOpReadVariableOp'twinned_q_network/q_network/dense1/bias*
_output_shapes	
:*
dtype0
¯
)twinned_q_network/q_network/output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*:
shared_name+)twinned_q_network/q_network/output/kernel
¨
=twinned_q_network/q_network/output/kernel/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network/output/kernel*
_output_shapes
:	*
dtype0
¦
'twinned_q_network/q_network/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'twinned_q_network/q_network/output/bias

;twinned_q_network/q_network/output/bias/Read/ReadVariableOpReadVariableOp'twinned_q_network/q_network/output/bias*
_output_shapes
:*
dtype0
°
)twinned_q_network/q_network/dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*:
shared_name+)twinned_q_network/q_network/dense2/kernel
©
=twinned_q_network/q_network/dense2/kernel/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network/dense2/kernel* 
_output_shapes
:
*
dtype0
§
'twinned_q_network/q_network/dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'twinned_q_network/q_network/dense2/bias
 
;twinned_q_network/q_network/dense2/bias/Read/ReadVariableOpReadVariableOp'twinned_q_network/q_network/dense2/bias*
_output_shapes	
:*
dtype0
³
+twinned_q_network/q_network_1/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	#*<
shared_name-+twinned_q_network/q_network_1/dense1/kernel
¬
?twinned_q_network/q_network_1/dense1/kernel/Read/ReadVariableOpReadVariableOp+twinned_q_network/q_network_1/dense1/kernel*
_output_shapes
:	#*
dtype0
«
)twinned_q_network/q_network_1/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)twinned_q_network/q_network_1/dense1/bias
¤
=twinned_q_network/q_network_1/dense1/bias/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network_1/dense1/bias*
_output_shapes	
:*
dtype0
³
+twinned_q_network/q_network_1/output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*<
shared_name-+twinned_q_network/q_network_1/output/kernel
¬
?twinned_q_network/q_network_1/output/kernel/Read/ReadVariableOpReadVariableOp+twinned_q_network/q_network_1/output/kernel*
_output_shapes
:	*
dtype0
ª
)twinned_q_network/q_network_1/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)twinned_q_network/q_network_1/output/bias
£
=twinned_q_network/q_network_1/output/bias/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network_1/output/bias*
_output_shapes
:*
dtype0
´
+twinned_q_network/q_network_1/dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*<
shared_name-+twinned_q_network/q_network_1/dense2/kernel
­
?twinned_q_network/q_network_1/dense2/kernel/Read/ReadVariableOpReadVariableOp+twinned_q_network/q_network_1/dense2/kernel* 
_output_shapes
:
*
dtype0
«
)twinned_q_network/q_network_1/dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)twinned_q_network/q_network_1/dense2/bias
¤
=twinned_q_network/q_network_1/dense2/bias/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network_1/dense2/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
³%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*î$
valueä$Bá$ BÚ$
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
trainable_variables
%metrics

&layers
'layer_metrics
	variables
(layer_regularization_losses
)non_trainable_variables
regularization_losses
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
trainable_variables
6metrics

7layers
8layer_metrics
	variables
9layer_regularization_losses
:non_trainable_variables
regularization_losses
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
trainable_variables
Gmetrics

Hlayers
Ilayer_metrics
	variables
Jlayer_regularization_losses
Knon_trainable_variables
regularization_losses
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

0
1
 
­
*trainable_variables
Lmetrics

Mlayers
Nlayer_metrics
+	variables
Olayer_regularization_losses
Pnon_trainable_variables
,regularization_losses

0
1

0
1
 
­
.trainable_variables
Qmetrics

Rlayers
Slayer_metrics
/	variables
Tlayer_regularization_losses
Unon_trainable_variables
0regularization_losses

0
1

0
1
 
­
2trainable_variables
Vmetrics

Wlayers
Xlayer_metrics
3	variables
Ylayer_regularization_losses
Znon_trainable_variables
4regularization_losses
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
;trainable_variables
[metrics

\layers
]layer_metrics
<	variables
^layer_regularization_losses
_non_trainable_variables
=regularization_losses

#0
$1

#0
$1
 
­
?trainable_variables
`metrics

alayers
blayer_metrics
@	variables
clayer_regularization_losses
dnon_trainable_variables
Aregularization_losses

!0
"1

!0
"1
 
­
Ctrainable_variables
emetrics

flayers
glayer_metrics
D	variables
hlayer_regularization_losses
inon_trainable_variables
Eregularization_losses
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
:ÿÿÿÿÿÿÿÿÿ#*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ#
å
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1)twinned_q_network/q_network/dense1/kernel'twinned_q_network/q_network/dense1/bias)twinned_q_network/q_network/dense2/kernel'twinned_q_network/q_network/dense2/bias)twinned_q_network/q_network/output/kernel'twinned_q_network/q_network/output/bias+twinned_q_network/q_network_1/dense1/kernel)twinned_q_network/q_network_1/dense1/bias+twinned_q_network/q_network_1/dense2/kernel)twinned_q_network/q_network_1/dense2/bias+twinned_q_network/q_network_1/output/kernel)twinned_q_network/q_network_1/output/bias*
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
"__inference_signature_wrapper_4974
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

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
GPU2 *0J 8 *&
f!R
__inference__traced_save_5152
©
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
GPU2 *0J 8 *)
f$R"
 __inference__traced_restore_5198«Ê
Ó

*__inference_q_network_1_layer_call_fn_4847
input_1
unknown:	#
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
E__inference_q_network_1_layer_call_and_return_conditional_losses_48292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ#: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
!
_user_specified_name	input_1
Þ
ü
E__inference_q_network_1_layer_call_and_return_conditional_losses_4829
input_1
dense1_4790:	#
dense1_4792:	
dense2_4807:

dense2_4809:	
output_4823:	
output_4825:
identity¢dense1/StatefulPartitionedCall¢dense2/StatefulPartitionedCall¢output/StatefulPartitionedCall
dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_4790dense1_4792*
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
@__inference_dense1_layer_call_and_return_conditional_losses_47892 
dense1/StatefulPartitionedCall«
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_4807dense2_4809*
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
@__inference_dense2_layer_call_and_return_conditional_losses_48062 
dense2/StatefulPartitionedCallª
output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_4823output_4825*
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
@__inference_output_layer_call_and_return_conditional_losses_48222 
output/StatefulPartitionedCallÞ
IdentityIdentity'output/StatefulPartitionedCall:output:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ#: : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
!
_user_specified_name	input_1
´

ô
@__inference_dense2_layer_call_and_return_conditional_losses_5073

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
÷:
÷	
 __inference__traced_restore_5198
file_prefixM
:assignvariableop_twinned_q_network_q_network_dense1_kernel:	#I
:assignvariableop_1_twinned_q_network_q_network_dense1_bias:	O
<assignvariableop_2_twinned_q_network_q_network_output_kernel:	H
:assignvariableop_3_twinned_q_network_q_network_output_bias:P
<assignvariableop_4_twinned_q_network_q_network_dense2_kernel:
I
:assignvariableop_5_twinned_q_network_q_network_dense2_bias:	Q
>assignvariableop_6_twinned_q_network_q_network_1_dense1_kernel:	#K
<assignvariableop_7_twinned_q_network_q_network_1_dense1_bias:	Q
>assignvariableop_8_twinned_q_network_q_network_1_output_kernel:	J
<assignvariableop_9_twinned_q_network_q_network_1_output_bias:S
?assignvariableop_10_twinned_q_network_q_network_1_dense2_kernel:
L
=assignvariableop_11_twinned_q_network_q_network_1_dense2_bias:	
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

Identity¹
AssignVariableOpAssignVariableOp:assignvariableop_twinned_q_network_q_network_dense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¿
AssignVariableOp_1AssignVariableOp:assignvariableop_1_twinned_q_network_q_network_dense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Á
AssignVariableOp_2AssignVariableOp<assignvariableop_2_twinned_q_network_q_network_output_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¿
AssignVariableOp_3AssignVariableOp:assignvariableop_3_twinned_q_network_q_network_output_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Á
AssignVariableOp_4AssignVariableOp<assignvariableop_4_twinned_q_network_q_network_dense2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¿
AssignVariableOp_5AssignVariableOp:assignvariableop_5_twinned_q_network_q_network_dense2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ã
AssignVariableOp_6AssignVariableOp>assignvariableop_6_twinned_q_network_q_network_1_dense1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Á
AssignVariableOp_7AssignVariableOp<assignvariableop_7_twinned_q_network_q_network_1_dense1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ã
AssignVariableOp_8AssignVariableOp>assignvariableop_8_twinned_q_network_q_network_1_output_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Á
AssignVariableOp_9AssignVariableOp<assignvariableop_9_twinned_q_network_q_network_1_output_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ç
AssignVariableOp_10AssignVariableOp?assignvariableop_10_twinned_q_network_q_network_1_dense2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Å
AssignVariableOp_11AssignVariableOp=assignvariableop_11_twinned_q_network_q_network_1_dense2_biasIdentity_11:output:0"/device:CPU:0*
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
¿
É
0__inference_twinned_q_network_layer_call_fn_4941
input_1
unknown:	#
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	#
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity

identity_1¢StatefulPartitionedCall
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
GPU2 *0J 8 *T
fORM
K__inference_twinned_q_network_layer_call_and_return_conditional_losses_49092
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
+:ÿÿÿÿÿÿÿÿÿ#: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
!
_user_specified_name	input_1
´

ô
@__inference_dense2_layer_call_and_return_conditional_losses_5014

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
@__inference_dense1_layer_call_and_return_conditional_losses_4789

inputs1
matmul_readvariableop_resource:	#.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	#*
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
:ÿÿÿÿÿÿÿÿÿ#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
Ð	
ò
@__inference_output_layer_call_and_return_conditional_losses_5092

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
@__inference_output_layer_call_and_return_conditional_losses_5033

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

»
"__inference_signature_wrapper_4974
input_1
unknown:	#
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:	#
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
__inference__wrapped_model_46712
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
+:ÿÿÿÿÿÿÿÿÿ#: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
!
_user_specified_name	input_1
`
§
__inference__wrapped_model_4671
input_1T
Atwinned_q_network_q_network_dense1_matmul_readvariableop_resource:	#Q
Btwinned_q_network_q_network_dense1_biasadd_readvariableop_resource:	U
Atwinned_q_network_q_network_dense2_matmul_readvariableop_resource:
Q
Btwinned_q_network_q_network_dense2_biasadd_readvariableop_resource:	T
Atwinned_q_network_q_network_output_matmul_readvariableop_resource:	P
Btwinned_q_network_q_network_output_biasadd_readvariableop_resource:V
Ctwinned_q_network_q_network_1_dense1_matmul_readvariableop_resource:	#S
Dtwinned_q_network_q_network_1_dense1_biasadd_readvariableop_resource:	W
Ctwinned_q_network_q_network_1_dense2_matmul_readvariableop_resource:
S
Dtwinned_q_network_q_network_1_dense2_biasadd_readvariableop_resource:	V
Ctwinned_q_network_q_network_1_output_matmul_readvariableop_resource:	R
Dtwinned_q_network_q_network_1_output_biasadd_readvariableop_resource:
identity

identity_1¢9twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOp¢8twinned_q_network/q_network/dense1/MatMul/ReadVariableOp¢9twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOp¢8twinned_q_network/q_network/dense2/MatMul/ReadVariableOp¢9twinned_q_network/q_network/output/BiasAdd/ReadVariableOp¢8twinned_q_network/q_network/output/MatMul/ReadVariableOp¢;twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOp¢:twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOp¢;twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOp¢:twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOp¢;twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOp¢:twinned_q_network/q_network_1/output/MatMul/ReadVariableOp÷
8twinned_q_network/q_network/dense1/MatMul/ReadVariableOpReadVariableOpAtwinned_q_network_q_network_dense1_matmul_readvariableop_resource*
_output_shapes
:	#*
dtype02:
8twinned_q_network/q_network/dense1/MatMul/ReadVariableOpÞ
)twinned_q_network/q_network/dense1/MatMulMatMulinput_1@twinned_q_network/q_network/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)twinned_q_network/q_network/dense1/MatMulö
9twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOpReadVariableOpBtwinned_q_network_q_network_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02;
9twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOp
*twinned_q_network/q_network/dense1/BiasAddBiasAdd3twinned_q_network/q_network/dense1/MatMul:product:0Atwinned_q_network/q_network/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*twinned_q_network/q_network/dense1/BiasAddÂ
'twinned_q_network/q_network/dense1/ReluRelu3twinned_q_network/q_network/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'twinned_q_network/q_network/dense1/Reluø
8twinned_q_network/q_network/dense2/MatMul/ReadVariableOpReadVariableOpAtwinned_q_network_q_network_dense2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02:
8twinned_q_network/q_network/dense2/MatMul/ReadVariableOp
)twinned_q_network/q_network/dense2/MatMulMatMul5twinned_q_network/q_network/dense1/Relu:activations:0@twinned_q_network/q_network/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)twinned_q_network/q_network/dense2/MatMulö
9twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOpReadVariableOpBtwinned_q_network_q_network_dense2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02;
9twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOp
*twinned_q_network/q_network/dense2/BiasAddBiasAdd3twinned_q_network/q_network/dense2/MatMul:product:0Atwinned_q_network/q_network/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*twinned_q_network/q_network/dense2/BiasAddÂ
'twinned_q_network/q_network/dense2/ReluRelu3twinned_q_network/q_network/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'twinned_q_network/q_network/dense2/Relu÷
8twinned_q_network/q_network/output/MatMul/ReadVariableOpReadVariableOpAtwinned_q_network_q_network_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02:
8twinned_q_network/q_network/output/MatMul/ReadVariableOp
)twinned_q_network/q_network/output/MatMulMatMul5twinned_q_network/q_network/dense2/Relu:activations:0@twinned_q_network/q_network/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)twinned_q_network/q_network/output/MatMulõ
9twinned_q_network/q_network/output/BiasAdd/ReadVariableOpReadVariableOpBtwinned_q_network_q_network_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9twinned_q_network/q_network/output/BiasAdd/ReadVariableOp
*twinned_q_network/q_network/output/BiasAddBiasAdd3twinned_q_network/q_network/output/MatMul:product:0Atwinned_q_network/q_network/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*twinned_q_network/q_network/output/BiasAddý
:twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOpReadVariableOpCtwinned_q_network_q_network_1_dense1_matmul_readvariableop_resource*
_output_shapes
:	#*
dtype02<
:twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOpä
+twinned_q_network/q_network_1/dense1/MatMulMatMulinput_1Btwinned_q_network/q_network_1/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+twinned_q_network/q_network_1/dense1/MatMulü
;twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOpReadVariableOpDtwinned_q_network_q_network_1_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOp
,twinned_q_network/q_network_1/dense1/BiasAddBiasAdd5twinned_q_network/q_network_1/dense1/MatMul:product:0Ctwinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,twinned_q_network/q_network_1/dense1/BiasAddÈ
)twinned_q_network/q_network_1/dense1/ReluRelu5twinned_q_network/q_network_1/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)twinned_q_network/q_network_1/dense1/Reluþ
:twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOpReadVariableOpCtwinned_q_network_q_network_1_dense2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02<
:twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOp
+twinned_q_network/q_network_1/dense2/MatMulMatMul7twinned_q_network/q_network_1/dense1/Relu:activations:0Btwinned_q_network/q_network_1/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+twinned_q_network/q_network_1/dense2/MatMulü
;twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOpReadVariableOpDtwinned_q_network_q_network_1_dense2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOp
,twinned_q_network/q_network_1/dense2/BiasAddBiasAdd5twinned_q_network/q_network_1/dense2/MatMul:product:0Ctwinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,twinned_q_network/q_network_1/dense2/BiasAddÈ
)twinned_q_network/q_network_1/dense2/ReluRelu5twinned_q_network/q_network_1/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)twinned_q_network/q_network_1/dense2/Reluý
:twinned_q_network/q_network_1/output/MatMul/ReadVariableOpReadVariableOpCtwinned_q_network_q_network_1_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02<
:twinned_q_network/q_network_1/output/MatMul/ReadVariableOp
+twinned_q_network/q_network_1/output/MatMulMatMul7twinned_q_network/q_network_1/dense2/Relu:activations:0Btwinned_q_network/q_network_1/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+twinned_q_network/q_network_1/output/MatMulû
;twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOpReadVariableOpDtwinned_q_network_q_network_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOp
,twinned_q_network/q_network_1/output/BiasAddBiasAdd5twinned_q_network/q_network_1/output/MatMul:product:0Ctwinned_q_network/q_network_1/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,twinned_q_network/q_network_1/output/BiasAddÝ
IdentityIdentity3twinned_q_network/q_network/output/BiasAdd:output:0:^twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/dense1/MatMul/ReadVariableOp:^twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/dense2/MatMul/ReadVariableOp:^twinned_q_network/q_network/output/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/output/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityã

Identity_1Identity5twinned_q_network/q_network_1/output/BiasAdd:output:0:^twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/dense1/MatMul/ReadVariableOp:^twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/dense2/MatMul/ReadVariableOp:^twinned_q_network/q_network/output/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/output/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ#: : : : : : : : : : : : 2v
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
:ÿÿÿÿÿÿÿÿÿ#
!
_user_specified_name	input_1


%__inference_dense1_layer_call_fn_4983

inputs
unknown:	#
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
@__inference_dense1_layer_call_and_return_conditional_losses_46862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs


%__inference_output_layer_call_fn_5023

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
@__inference_output_layer_call_and_return_conditional_losses_47192
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
Ü
ú
C__inference_q_network_layer_call_and_return_conditional_losses_4726
input_1
dense1_4687:	#
dense1_4689:	
dense2_4704:

dense2_4706:	
output_4720:	
output_4722:
identity¢dense1/StatefulPartitionedCall¢dense2/StatefulPartitionedCall¢output/StatefulPartitionedCall
dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_4687dense1_4689*
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
@__inference_dense1_layer_call_and_return_conditional_losses_46862 
dense1/StatefulPartitionedCall«
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_4704dense2_4706*
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
@__inference_dense2_layer_call_and_return_conditional_losses_47032 
dense2/StatefulPartitionedCallª
output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_4720output_4722*
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
@__inference_output_layer_call_and_return_conditional_losses_47192 
output/StatefulPartitionedCallÞ
IdentityIdentity'output/StatefulPartitionedCall:output:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ#: : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
!
_user_specified_name	input_1


%__inference_dense2_layer_call_fn_5062

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
@__inference_dense2_layer_call_and_return_conditional_losses_48062
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
@__inference_output_layer_call_and_return_conditional_losses_4719

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
%__inference_dense2_layer_call_fn_5003

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
@__inference_dense2_layer_call_and_return_conditional_losses_47032
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
°

ó
@__inference_dense1_layer_call_and_return_conditional_losses_5053

inputs1
matmul_readvariableop_resource:	#.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	#*
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
:ÿÿÿÿÿÿÿÿÿ#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs


%__inference_output_layer_call_fn_5082

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
@__inference_output_layer_call_and_return_conditional_losses_48222
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
¾)
â
__inference__traced_save_5152
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
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Dsavev2_twinned_q_network_q_network_dense1_kernel_read_readvariableopBsavev2_twinned_q_network_q_network_dense1_bias_read_readvariableopDsavev2_twinned_q_network_q_network_output_kernel_read_readvariableopBsavev2_twinned_q_network_q_network_output_bias_read_readvariableopDsavev2_twinned_q_network_q_network_dense2_kernel_read_readvariableopBsavev2_twinned_q_network_q_network_dense2_bias_read_readvariableopFsavev2_twinned_q_network_q_network_1_dense1_kernel_read_readvariableopDsavev2_twinned_q_network_q_network_1_dense1_bias_read_readvariableopFsavev2_twinned_q_network_q_network_1_output_kernel_read_readvariableopDsavev2_twinned_q_network_q_network_1_output_bias_read_readvariableopFsavev2_twinned_q_network_q_network_1_dense2_kernel_read_readvariableopDsavev2_twinned_q_network_q_network_1_dense2_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
p: :	#::	::
::	#::	::
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	#:!
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
:	#:!
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
°

ó
@__inference_dense1_layer_call_and_return_conditional_losses_4994

inputs1
matmul_readvariableop_resource:	#.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	#*
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
:ÿÿÿÿÿÿÿÿÿ#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs


%__inference_dense1_layer_call_fn_5042

inputs
unknown:	#
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
@__inference_dense1_layer_call_and_return_conditional_losses_47892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
Ð	
ò
@__inference_output_layer_call_and_return_conditional_losses_4822

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
´

ô
@__inference_dense2_layer_call_and_return_conditional_losses_4703

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
è
Ý
K__inference_twinned_q_network_layer_call_and_return_conditional_losses_4909
input_1!
q_network_4881:	#
q_network_4883:	"
q_network_4885:

q_network_4887:	!
q_network_4889:	
q_network_4891:#
q_network_1_4894:	#
q_network_1_4896:	$
q_network_1_4898:

q_network_1_4900:	#
q_network_1_4902:	
q_network_1_4904:
identity

identity_1¢!q_network/StatefulPartitionedCall¢#q_network_1/StatefulPartitionedCallá
!q_network/StatefulPartitionedCallStatefulPartitionedCallinput_1q_network_4881q_network_4883q_network_4885q_network_4887q_network_4889q_network_4891*
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
GPU2 *0J 8 *L
fGRE
C__inference_q_network_layer_call_and_return_conditional_losses_47262#
!q_network/StatefulPartitionedCalló
#q_network_1/StatefulPartitionedCallStatefulPartitionedCallinput_1q_network_1_4894q_network_1_4896q_network_1_4898q_network_1_4900q_network_1_4902q_network_1_4904*
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
E__inference_q_network_1_layer_call_and_return_conditional_losses_48292%
#q_network_1/StatefulPartitionedCallÈ
IdentityIdentity*q_network/StatefulPartitionedCall:output:0"^q_network/StatefulPartitionedCall$^q_network_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÎ

Identity_1Identity,q_network_1/StatefulPartitionedCall:output:0"^q_network/StatefulPartitionedCall$^q_network_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ#: : : : : : : : : : : : 2F
!q_network/StatefulPartitionedCall!q_network/StatefulPartitionedCall2J
#q_network_1/StatefulPartitionedCall#q_network_1/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
!
_user_specified_name	input_1
´

ô
@__inference_dense2_layer_call_and_return_conditional_losses_4806

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
@__inference_dense1_layer_call_and_return_conditional_losses_4686

inputs1
matmul_readvariableop_resource:	#.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	#*
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
:ÿÿÿÿÿÿÿÿÿ#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
Ï

(__inference_q_network_layer_call_fn_4744
input_1
unknown:	#
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
identity¢StatefulPartitionedCall­
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
GPU2 *0J 8 *L
fGRE
C__inference_q_network_layer_call_and_return_conditional_losses_47262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ#: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
!
_user_specified_name	input_1"ÌL
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
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ#<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ<
output_20
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¨¶

Q1
Q2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
j_default_save_signature
k__call__
*l&call_and_return_all_conditional_losses"¯
_tf_keras_model{"name": "twinned_q_network", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "TwinnedQNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 35]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "TwinnedQNetwork"}}
Û
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
*n&call_and_return_all_conditional_losses"
_tf_keras_modelÿ{"name": "q_network", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "QNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 35]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "QNetwork"}}
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
_tf_keras_model{"name": "q_network_1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "QNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 35]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "QNetwork"}}
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
trainable_variables
%metrics

&layers
'layer_metrics
	variables
(layer_regularization_losses
)non_trainable_variables
regularization_losses
k__call__
j_default_save_signature
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
*trainable_variables
+	variables
,regularization_losses
-	keras_api
r__call__
*s&call_and_return_all_conditional_losses"£
_tf_keras_layer{"name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 35}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 35]}}
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
trainable_variables
6metrics

7layers
8layer_metrics
	variables
9layer_regularization_losses
:non_trainable_variables
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
;trainable_variables
<	variables
=regularization_losses
>	keras_api
x__call__
*y&call_and_return_all_conditional_losses"§
_tf_keras_layer{"name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 35}}, "shared_object_id": 15}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 35]}}
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
trainable_variables
Gmetrics

Hlayers
Ilayer_metrics
	variables
Jlayer_regularization_losses
Knon_trainable_variables
regularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
<::	#2)twinned_q_network/q_network/dense1/kernel
6:42'twinned_q_network/q_network/dense1/bias
<::	2)twinned_q_network/q_network/output/kernel
5:32'twinned_q_network/q_network/output/bias
=:;
2)twinned_q_network/q_network/dense2/kernel
6:42'twinned_q_network/q_network/dense2/bias
>:<	#2+twinned_q_network/q_network_1/dense1/kernel
8:62)twinned_q_network/q_network_1/dense1/bias
>:<	2+twinned_q_network/q_network_1/output/kernel
7:52)twinned_q_network/q_network_1/output/bias
?:=
2+twinned_q_network/q_network_1/dense2/kernel
8:62)twinned_q_network/q_network_1/dense2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
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
*trainable_variables
Lmetrics

Mlayers
Nlayer_metrics
+	variables
Olayer_regularization_losses
Pnon_trainable_variables
,regularization_losses
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
.trainable_variables
Qmetrics

Rlayers
Slayer_metrics
/	variables
Tlayer_regularization_losses
Unon_trainable_variables
0regularization_losses
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
2trainable_variables
Vmetrics

Wlayers
Xlayer_metrics
3	variables
Ylayer_regularization_losses
Znon_trainable_variables
4regularization_losses
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
trackable_dict_wrapper
 "
trackable_list_wrapper
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
;trainable_variables
[metrics

\layers
]layer_metrics
<	variables
^layer_regularization_losses
_non_trainable_variables
=regularization_losses
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
?trainable_variables
`metrics

alayers
blayer_metrics
@	variables
clayer_regularization_losses
dnon_trainable_variables
Aregularization_losses
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
Ctrainable_variables
emetrics

flayers
glayer_metrics
D	variables
hlayer_regularization_losses
inon_trainable_variables
Eregularization_losses
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
 "
trackable_list_wrapper
Ý2Ú
__inference__wrapped_model_4671¶
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
input_1ÿÿÿÿÿÿÿÿÿ#
ù2ö
0__inference_twinned_q_network_layer_call_fn_4941Á
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
input_1ÿÿÿÿÿÿÿÿÿ#
2
K__inference_twinned_q_network_layer_call_and_return_conditional_losses_4909Á
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
input_1ÿÿÿÿÿÿÿÿÿ#
ñ2î
(__inference_q_network_layer_call_fn_4744Á
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
input_1ÿÿÿÿÿÿÿÿÿ#
2
C__inference_q_network_layer_call_and_return_conditional_losses_4726Á
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
input_1ÿÿÿÿÿÿÿÿÿ#
ó2ð
*__inference_q_network_1_layer_call_fn_4847Á
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
input_1ÿÿÿÿÿÿÿÿÿ#
2
E__inference_q_network_1_layer_call_and_return_conditional_losses_4829Á
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
input_1ÿÿÿÿÿÿÿÿÿ#
ÉBÆ
"__inference_signature_wrapper_4974input_1"
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
%__inference_dense1_layer_call_fn_4983¢
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
@__inference_dense1_layer_call_and_return_conditional_losses_4994¢
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
%__inference_dense2_layer_call_fn_5003¢
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
@__inference_dense2_layer_call_and_return_conditional_losses_5014¢
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
%__inference_output_layer_call_fn_5023¢
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
@__inference_output_layer_call_and_return_conditional_losses_5033¢
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
%__inference_dense1_layer_call_fn_5042¢
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
@__inference_dense1_layer_call_and_return_conditional_losses_5053¢
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
%__inference_dense2_layer_call_fn_5062¢
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
@__inference_dense2_layer_call_and_return_conditional_losses_5073¢
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
%__inference_output_layer_call_fn_5082¢
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
@__inference_output_layer_call_and_return_conditional_losses_5092¢
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
__inference__wrapped_model_4671¥ #$!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ#
ª "cª`
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿ¡
@__inference_dense1_layer_call_and_return_conditional_losses_4994]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ#
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¡
@__inference_dense1_layer_call_and_return_conditional_losses_5053] /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ#
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 y
%__inference_dense1_layer_call_fn_4983P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ#
ª "ÿÿÿÿÿÿÿÿÿy
%__inference_dense1_layer_call_fn_5042P /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ#
ª "ÿÿÿÿÿÿÿÿÿ¢
@__inference_dense2_layer_call_and_return_conditional_losses_5014^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¢
@__inference_dense2_layer_call_and_return_conditional_losses_5073^#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 z
%__inference_dense2_layer_call_fn_5003Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿz
%__inference_dense2_layer_call_fn_5062Q#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¡
@__inference_output_layer_call_and_return_conditional_losses_5033]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¡
@__inference_output_layer_call_and_return_conditional_losses_5092]!"0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
%__inference_output_layer_call_fn_5023P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿy
%__inference_output_layer_call_fn_5082P!"0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
E__inference_q_network_1_layer_call_and_return_conditional_losses_4829a #$!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ#
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_q_network_1_layer_call_fn_4847T #$!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ#
ª "ÿÿÿÿÿÿÿÿÿ¨
C__inference_q_network_layer_call_and_return_conditional_losses_4726a0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ#
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_q_network_layer_call_fn_4744T0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ#
ª "ÿÿÿÿÿÿÿÿÿ×
"__inference_signature_wrapper_4974° #$!";¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ#"cª`
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿÝ
K__inference_twinned_q_network_layer_call_and_return_conditional_losses_4909 #$!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ#
ª "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 ³
0__inference_twinned_q_network_layer_call_fn_4941 #$!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ#
ª "=¢:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ