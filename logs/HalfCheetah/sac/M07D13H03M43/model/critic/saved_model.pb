▄ю
│Г
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
delete_old_dirsbool(И
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
╛
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
executor_typestring И
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718уж
п
)twinned_q_network/q_network/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*:
shared_name+)twinned_q_network/q_network/dense1/kernel
и
=twinned_q_network/q_network/dense1/kernel/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network/dense1/kernel*
_output_shapes
:	А*
dtype0
з
'twinned_q_network/q_network/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*8
shared_name)'twinned_q_network/q_network/dense1/bias
а
;twinned_q_network/q_network/dense1/bias/Read/ReadVariableOpReadVariableOp'twinned_q_network/q_network/dense1/bias*
_output_shapes	
:А*
dtype0
░
)twinned_q_network/q_network/dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*:
shared_name+)twinned_q_network/q_network/dense2/kernel
й
=twinned_q_network/q_network/dense2/kernel/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network/dense2/kernel* 
_output_shapes
:
АА*
dtype0
з
'twinned_q_network/q_network/dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*8
shared_name)'twinned_q_network/q_network/dense2/bias
а
;twinned_q_network/q_network/dense2/bias/Read/ReadVariableOpReadVariableOp'twinned_q_network/q_network/dense2/bias*
_output_shapes	
:А*
dtype0
п
)twinned_q_network/q_network/output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*:
shared_name+)twinned_q_network/q_network/output/kernel
и
=twinned_q_network/q_network/output/kernel/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network/output/kernel*
_output_shapes
:	А*
dtype0
ж
'twinned_q_network/q_network/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'twinned_q_network/q_network/output/bias
Я
;twinned_q_network/q_network/output/bias/Read/ReadVariableOpReadVariableOp'twinned_q_network/q_network/output/bias*
_output_shapes
:*
dtype0
│
+twinned_q_network/q_network_1/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*<
shared_name-+twinned_q_network/q_network_1/dense1/kernel
м
?twinned_q_network/q_network_1/dense1/kernel/Read/ReadVariableOpReadVariableOp+twinned_q_network/q_network_1/dense1/kernel*
_output_shapes
:	А*
dtype0
л
)twinned_q_network/q_network_1/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*:
shared_name+)twinned_q_network/q_network_1/dense1/bias
д
=twinned_q_network/q_network_1/dense1/bias/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network_1/dense1/bias*
_output_shapes	
:А*
dtype0
┤
+twinned_q_network/q_network_1/dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*<
shared_name-+twinned_q_network/q_network_1/dense2/kernel
н
?twinned_q_network/q_network_1/dense2/kernel/Read/ReadVariableOpReadVariableOp+twinned_q_network/q_network_1/dense2/kernel* 
_output_shapes
:
АА*
dtype0
л
)twinned_q_network/q_network_1/dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*:
shared_name+)twinned_q_network/q_network_1/dense2/bias
д
=twinned_q_network/q_network_1/dense2/bias/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network_1/dense2/bias*
_output_shapes	
:А*
dtype0
│
+twinned_q_network/q_network_1/output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*<
shared_name-+twinned_q_network/q_network_1/output/kernel
м
?twinned_q_network/q_network_1/output/kernel/Read/ReadVariableOpReadVariableOp+twinned_q_network/q_network_1/output/kernel*
_output_shapes
:	А*
dtype0
к
)twinned_q_network/q_network_1/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)twinned_q_network/q_network_1/output/bias
г
=twinned_q_network/q_network_1/output/bias/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network_1/output/bias*
_output_shapes
:*
dtype0

NoOpNoOp
╗$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ў#
valueь#Bщ# Bт#
Б
Q1
Q2
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
Е
	hidden_units


dense1

dense2
out
	variables
regularization_losses
trainable_variables
	keras_api
Е
hidden_units

dense1

dense2
out
	variables
regularization_losses
trainable_variables
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
н
	variables
%layer_metrics
&metrics
'layer_regularization_losses

(layers
regularization_losses
)non_trainable_variables
trainable_variables
 
 
h

kernel
bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
h

kernel
bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
h

kernel
bias
2	variables
3regularization_losses
4trainable_variables
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
н
	variables
6layer_metrics
7metrics
8layer_regularization_losses

9layers
regularization_losses
:non_trainable_variables
trainable_variables
 
h

kernel
 bias
;	variables
<regularization_losses
=trainable_variables
>	keras_api
h

!kernel
"bias
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
h

#kernel
$bias
C	variables
Dregularization_losses
Etrainable_variables
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
н
	variables
Glayer_metrics
Hmetrics
Ilayer_regularization_losses

Jlayers
regularization_losses
Knon_trainable_variables
trainable_variables
ec
VARIABLE_VALUE)twinned_q_network/q_network/dense1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'twinned_q_network/q_network/dense1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)twinned_q_network/q_network/dense2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'twinned_q_network/q_network/dense2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)twinned_q_network/q_network/output/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'twinned_q_network/q_network/output/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+twinned_q_network/q_network_1/dense1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)twinned_q_network/q_network_1/dense1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+twinned_q_network/q_network_1/dense2/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)twinned_q_network/q_network_1/dense2/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+twinned_q_network/q_network_1/output/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)twinned_q_network/q_network_1/output/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
 

0
1
 

0
1
н
*	variables
Llayer_metrics
Mmetrics
Nlayer_regularization_losses

Olayers
+regularization_losses
Pnon_trainable_variables
,trainable_variables

0
1
 

0
1
н
.	variables
Qlayer_metrics
Rmetrics
Slayer_regularization_losses

Tlayers
/regularization_losses
Unon_trainable_variables
0trainable_variables

0
1
 

0
1
н
2	variables
Vlayer_metrics
Wmetrics
Xlayer_regularization_losses

Ylayers
3regularization_losses
Znon_trainable_variables
4trainable_variables
 
 
 


0
1
2
 

0
 1
 

0
 1
н
;	variables
[layer_metrics
\metrics
]layer_regularization_losses

^layers
<regularization_losses
_non_trainable_variables
=trainable_variables

!0
"1
 

!0
"1
н
?	variables
`layer_metrics
ametrics
blayer_regularization_losses

clayers
@regularization_losses
dnon_trainable_variables
Atrainable_variables

#0
$1
 

#0
$1
н
C	variables
elayer_metrics
fmetrics
glayer_regularization_losses

hlayers
Dregularization_losses
inon_trainable_variables
Etrainable_variables
 
 
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
z
serving_default_input_1Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
ш
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1)twinned_q_network/q_network/dense1/kernel'twinned_q_network/q_network/dense1/bias)twinned_q_network/q_network/dense2/kernel'twinned_q_network/q_network/dense2/bias)twinned_q_network/q_network/output/kernel'twinned_q_network/q_network/output/bias+twinned_q_network/q_network_1/dense1/kernel)twinned_q_network/q_network_1/dense1/bias+twinned_q_network/q_network_1/dense2/kernel)twinned_q_network/q_network_1/dense2/bias+twinned_q_network/q_network_1/output/kernel)twinned_q_network/q_network_1/output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         *.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *.
f)R'
%__inference_signature_wrapper_3338693
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
б
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename=twinned_q_network/q_network/dense1/kernel/Read/ReadVariableOp;twinned_q_network/q_network/dense1/bias/Read/ReadVariableOp=twinned_q_network/q_network/dense2/kernel/Read/ReadVariableOp;twinned_q_network/q_network/dense2/bias/Read/ReadVariableOp=twinned_q_network/q_network/output/kernel/Read/ReadVariableOp;twinned_q_network/q_network/output/bias/Read/ReadVariableOp?twinned_q_network/q_network_1/dense1/kernel/Read/ReadVariableOp=twinned_q_network/q_network_1/dense1/bias/Read/ReadVariableOp?twinned_q_network/q_network_1/dense2/kernel/Read/ReadVariableOp=twinned_q_network/q_network_1/dense2/bias/Read/ReadVariableOp?twinned_q_network/q_network_1/output/kernel/Read/ReadVariableOp=twinned_q_network/q_network_1/output/bias/Read/ReadVariableOpConst*
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
GPU2 *0J 8В *)
f$R"
 __inference__traced_save_3338871
м
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename)twinned_q_network/q_network/dense1/kernel'twinned_q_network/q_network/dense1/bias)twinned_q_network/q_network/dense2/kernel'twinned_q_network/q_network/dense2/bias)twinned_q_network/q_network/output/kernel'twinned_q_network/q_network/output/bias+twinned_q_network/q_network_1/dense1/kernel)twinned_q_network/q_network_1/dense1/bias+twinned_q_network/q_network_1/dense2/kernel)twinned_q_network/q_network_1/dense2/bias+twinned_q_network/q_network_1/output/kernel)twinned_q_network/q_network_1/output/bias*
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
GPU2 *0J 8В *,
f'R%
#__inference__traced_restore_3338917╒╩
М
П
F__inference_q_network_layer_call_and_return_conditional_losses_3338445
input_1!
dense1_3338406:	А
dense1_3338408:	А"
dense2_3338423:
АА
dense2_3338425:	А!
output_3338439:	А
output_3338441:
identityИвdense1/StatefulPartitionedCallвdense2/StatefulPartitionedCallвoutput/StatefulPartitionedCallФ
dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_3338406dense1_3338408*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_33384052 
dense1/StatefulPartitionedCall┤
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_3338423dense2_3338425*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_33384222 
dense2/StatefulPartitionedCall│
output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_3338439output_3338441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_33384382 
output/StatefulPartitionedCall▐
IdentityIdentity'output/StatefulPartitionedCall:output:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
О
С
H__inference_q_network_1_layer_call_and_return_conditional_losses_3338548
input_1!
dense1_3338509:	А
dense1_3338511:	А"
dense2_3338526:
АА
dense2_3338528:	А!
output_3338542:	А
output_3338544:
identityИвdense1/StatefulPartitionedCallвdense2/StatefulPartitionedCallвoutput/StatefulPartitionedCallФ
dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_3338509dense1_3338511*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_33385082 
dense1/StatefulPartitionedCall┤
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_3338526dense2_3338528*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_33385252 
dense2/StatefulPartitionedCall│
output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_3338542output_3338544*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_33385412 
output/StatefulPartitionedCall▐
IdentityIdentity'output/StatefulPartitionedCall:output:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
┼
╠
3__inference_twinned_q_network_layer_call_fn_3338660
input_1
unknown:	А
	unknown_0:	А
	unknown_1:
АА
	unknown_2:	А
	unknown_3:	А
	unknown_4:
	unknown_5:	А
	unknown_6:	А
	unknown_7:
АА
	unknown_8:	А
	unknown_9:	А

unknown_10:
identity

identity_1ИвStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         *.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_twinned_q_network_layer_call_and_return_conditional_losses_33386282
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
╙	
ї
C__inference_output_layer_call_and_return_conditional_losses_3338438

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
│

Ў
C__inference_dense1_layer_call_and_return_conditional_losses_3338713

inputs1
matmul_readvariableop_resource:	А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╣
Д
N__inference_twinned_q_network_layer_call_and_return_conditional_losses_3338628
input_1$
q_network_3338600:	А 
q_network_3338602:	А%
q_network_3338604:
АА 
q_network_3338606:	А$
q_network_3338608:	А
q_network_3338610:&
q_network_1_3338613:	А"
q_network_1_3338615:	А'
q_network_1_3338617:
АА"
q_network_1_3338619:	А&
q_network_1_3338621:	А!
q_network_1_3338623:
identity

identity_1Ив!q_network/StatefulPartitionedCallв#q_network_1/StatefulPartitionedCallЎ
!q_network/StatefulPartitionedCallStatefulPartitionedCallinput_1q_network_3338600q_network_3338602q_network_3338604q_network_3338606q_network_3338608q_network_3338610*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_q_network_layer_call_and_return_conditional_losses_33384452#
!q_network/StatefulPartitionedCallИ
#q_network_1/StatefulPartitionedCallStatefulPartitionedCallinput_1q_network_1_3338613q_network_1_3338615q_network_1_3338617q_network_1_3338619q_network_1_3338621q_network_1_3338623*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_q_network_1_layer_call_and_return_conditional_losses_33385482%
#q_network_1/StatefulPartitionedCall╚
IdentityIdentity*q_network/StatefulPartitionedCall:output:0"^q_network/StatefulPartitionedCall$^q_network_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity╬

Identity_1Identity,q_network_1/StatefulPartitionedCall:output:0"^q_network/StatefulPartitionedCall$^q_network_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : : : 2F
!q_network/StatefulPartitionedCall!q_network/StatefulPartitionedCall2J
#q_network_1/StatefulPartitionedCall#q_network_1/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
╙	
ї
C__inference_output_layer_call_and_return_conditional_losses_3338752

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
в
Ч
(__inference_dense1_layer_call_fn_3338702

inputs
unknown:	А
	unknown_0:	А
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_33384052
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
е
Ш
(__inference_dense2_layer_call_fn_3338722

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_33384222
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
б
Ц
(__inference_output_layer_call_fn_3338742

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_33384382
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╒
Л
+__inference_q_network_layer_call_fn_3338463
input_1
unknown:	А
	unknown_0:	А
	unknown_1:
АА
	unknown_2:	А
	unknown_3:	А
	unknown_4:
identityИвStatefulPartitionedCall░
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_q_network_layer_call_and_return_conditional_losses_33384452
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
╙	
ї
C__inference_output_layer_call_and_return_conditional_losses_3338541

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
в
Ч
(__inference_dense1_layer_call_fn_3338761

inputs
unknown:	А
	unknown_0:	А
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_33385082
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
│

Ў
C__inference_dense1_layer_call_and_return_conditional_losses_3338772

inputs1
matmul_readvariableop_resource:	А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
б`
к
"__inference__wrapped_model_3338390
input_1T
Atwinned_q_network_q_network_dense1_matmul_readvariableop_resource:	АQ
Btwinned_q_network_q_network_dense1_biasadd_readvariableop_resource:	АU
Atwinned_q_network_q_network_dense2_matmul_readvariableop_resource:
ААQ
Btwinned_q_network_q_network_dense2_biasadd_readvariableop_resource:	АT
Atwinned_q_network_q_network_output_matmul_readvariableop_resource:	АP
Btwinned_q_network_q_network_output_biasadd_readvariableop_resource:V
Ctwinned_q_network_q_network_1_dense1_matmul_readvariableop_resource:	АS
Dtwinned_q_network_q_network_1_dense1_biasadd_readvariableop_resource:	АW
Ctwinned_q_network_q_network_1_dense2_matmul_readvariableop_resource:
ААS
Dtwinned_q_network_q_network_1_dense2_biasadd_readvariableop_resource:	АV
Ctwinned_q_network_q_network_1_output_matmul_readvariableop_resource:	АR
Dtwinned_q_network_q_network_1_output_biasadd_readvariableop_resource:
identity

identity_1Ив9twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOpв8twinned_q_network/q_network/dense1/MatMul/ReadVariableOpв9twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOpв8twinned_q_network/q_network/dense2/MatMul/ReadVariableOpв9twinned_q_network/q_network/output/BiasAdd/ReadVariableOpв8twinned_q_network/q_network/output/MatMul/ReadVariableOpв;twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOpв:twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOpв;twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOpв:twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOpв;twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOpв:twinned_q_network/q_network_1/output/MatMul/ReadVariableOpў
8twinned_q_network/q_network/dense1/MatMul/ReadVariableOpReadVariableOpAtwinned_q_network_q_network_dense1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02:
8twinned_q_network/q_network/dense1/MatMul/ReadVariableOp▐
)twinned_q_network/q_network/dense1/MatMulMatMulinput_1@twinned_q_network/q_network/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2+
)twinned_q_network/q_network/dense1/MatMulЎ
9twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOpReadVariableOpBtwinned_q_network_q_network_dense1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOpО
*twinned_q_network/q_network/dense1/BiasAddBiasAdd3twinned_q_network/q_network/dense1/MatMul:product:0Atwinned_q_network/q_network/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*twinned_q_network/q_network/dense1/BiasAdd┬
'twinned_q_network/q_network/dense1/ReluRelu3twinned_q_network/q_network/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:         А2)
'twinned_q_network/q_network/dense1/Relu°
8twinned_q_network/q_network/dense2/MatMul/ReadVariableOpReadVariableOpAtwinned_q_network_q_network_dense2_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02:
8twinned_q_network/q_network/dense2/MatMul/ReadVariableOpМ
)twinned_q_network/q_network/dense2/MatMulMatMul5twinned_q_network/q_network/dense1/Relu:activations:0@twinned_q_network/q_network/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2+
)twinned_q_network/q_network/dense2/MatMulЎ
9twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOpReadVariableOpBtwinned_q_network_q_network_dense2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOpО
*twinned_q_network/q_network/dense2/BiasAddBiasAdd3twinned_q_network/q_network/dense2/MatMul:product:0Atwinned_q_network/q_network/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*twinned_q_network/q_network/dense2/BiasAdd┬
'twinned_q_network/q_network/dense2/ReluRelu3twinned_q_network/q_network/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:         А2)
'twinned_q_network/q_network/dense2/Reluў
8twinned_q_network/q_network/output/MatMul/ReadVariableOpReadVariableOpAtwinned_q_network_q_network_output_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02:
8twinned_q_network/q_network/output/MatMul/ReadVariableOpЛ
)twinned_q_network/q_network/output/MatMulMatMul5twinned_q_network/q_network/dense2/Relu:activations:0@twinned_q_network/q_network/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2+
)twinned_q_network/q_network/output/MatMulї
9twinned_q_network/q_network/output/BiasAdd/ReadVariableOpReadVariableOpBtwinned_q_network_q_network_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9twinned_q_network/q_network/output/BiasAdd/ReadVariableOpН
*twinned_q_network/q_network/output/BiasAddBiasAdd3twinned_q_network/q_network/output/MatMul:product:0Atwinned_q_network/q_network/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2,
*twinned_q_network/q_network/output/BiasAdd¤
:twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOpReadVariableOpCtwinned_q_network_q_network_1_dense1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02<
:twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOpф
+twinned_q_network/q_network_1/dense1/MatMulMatMulinput_1Btwinned_q_network/q_network_1/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2-
+twinned_q_network/q_network_1/dense1/MatMul№
;twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOpReadVariableOpDtwinned_q_network_q_network_1_dense1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02=
;twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOpЦ
,twinned_q_network/q_network_1/dense1/BiasAddBiasAdd5twinned_q_network/q_network_1/dense1/MatMul:product:0Ctwinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2.
,twinned_q_network/q_network_1/dense1/BiasAdd╚
)twinned_q_network/q_network_1/dense1/ReluRelu5twinned_q_network/q_network_1/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:         А2+
)twinned_q_network/q_network_1/dense1/Relu■
:twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOpReadVariableOpCtwinned_q_network_q_network_1_dense2_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02<
:twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOpФ
+twinned_q_network/q_network_1/dense2/MatMulMatMul7twinned_q_network/q_network_1/dense1/Relu:activations:0Btwinned_q_network/q_network_1/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2-
+twinned_q_network/q_network_1/dense2/MatMul№
;twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOpReadVariableOpDtwinned_q_network_q_network_1_dense2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02=
;twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOpЦ
,twinned_q_network/q_network_1/dense2/BiasAddBiasAdd5twinned_q_network/q_network_1/dense2/MatMul:product:0Ctwinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2.
,twinned_q_network/q_network_1/dense2/BiasAdd╚
)twinned_q_network/q_network_1/dense2/ReluRelu5twinned_q_network/q_network_1/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:         А2+
)twinned_q_network/q_network_1/dense2/Relu¤
:twinned_q_network/q_network_1/output/MatMul/ReadVariableOpReadVariableOpCtwinned_q_network_q_network_1_output_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02<
:twinned_q_network/q_network_1/output/MatMul/ReadVariableOpУ
+twinned_q_network/q_network_1/output/MatMulMatMul7twinned_q_network/q_network_1/dense2/Relu:activations:0Btwinned_q_network/q_network_1/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2-
+twinned_q_network/q_network_1/output/MatMul√
;twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOpReadVariableOpDtwinned_q_network_q_network_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOpХ
,twinned_q_network/q_network_1/output/BiasAddBiasAdd5twinned_q_network/q_network_1/output/MatMul:product:0Ctwinned_q_network/q_network_1/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2.
,twinned_q_network/q_network_1/output/BiasAdd▌
IdentityIdentity3twinned_q_network/q_network/output/BiasAdd:output:0:^twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/dense1/MatMul/ReadVariableOp:^twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/dense2/MatMul/ReadVariableOp:^twinned_q_network/q_network/output/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/output/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identityу

Identity_1Identity5twinned_q_network/q_network_1/output/BiasAdd:output:0:^twinned_q_network/q_network/dense1/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/dense1/MatMul/ReadVariableOp:^twinned_q_network/q_network/dense2/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/dense2/MatMul/ReadVariableOp:^twinned_q_network/q_network/output/BiasAdd/ReadVariableOp9^twinned_q_network/q_network/output/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/dense1/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/dense1/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/dense2/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/dense2/MatMul/ReadVariableOp<^twinned_q_network/q_network_1/output/BiasAdd/ReadVariableOp;^twinned_q_network/q_network_1/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : : : 2v
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
:         
!
_user_specified_name	input_1
╔(
х
 __inference__traced_save_3338871
file_prefixH
Dsavev2_twinned_q_network_q_network_dense1_kernel_read_readvariableopF
Bsavev2_twinned_q_network_q_network_dense1_bias_read_readvariableopH
Dsavev2_twinned_q_network_q_network_dense2_kernel_read_readvariableopF
Bsavev2_twinned_q_network_q_network_dense2_bias_read_readvariableopH
Dsavev2_twinned_q_network_q_network_output_kernel_read_readvariableopF
Bsavev2_twinned_q_network_q_network_output_bias_read_readvariableopJ
Fsavev2_twinned_q_network_q_network_1_dense1_kernel_read_readvariableopH
Dsavev2_twinned_q_network_q_network_1_dense1_bias_read_readvariableopJ
Fsavev2_twinned_q_network_q_network_1_dense2_kernel_read_readvariableopH
Dsavev2_twinned_q_network_q_network_1_dense2_bias_read_readvariableopJ
Fsavev2_twinned_q_network_q_network_1_output_kernel_read_readvariableopH
Dsavev2_twinned_q_network_q_network_1_output_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameГ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Х
valueЛBИB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesв
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesО
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Dsavev2_twinned_q_network_q_network_dense1_kernel_read_readvariableopBsavev2_twinned_q_network_q_network_dense1_bias_read_readvariableopDsavev2_twinned_q_network_q_network_dense2_kernel_read_readvariableopBsavev2_twinned_q_network_q_network_dense2_bias_read_readvariableopDsavev2_twinned_q_network_q_network_output_kernel_read_readvariableopBsavev2_twinned_q_network_q_network_output_bias_read_readvariableopFsavev2_twinned_q_network_q_network_1_dense1_kernel_read_readvariableopDsavev2_twinned_q_network_q_network_1_dense1_bias_read_readvariableopFsavev2_twinned_q_network_q_network_1_dense2_kernel_read_readvariableopDsavev2_twinned_q_network_q_network_1_dense2_bias_read_readvariableopFsavev2_twinned_q_network_q_network_1_output_kernel_read_readvariableopDsavev2_twinned_q_network_q_network_1_output_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*Г
_input_shapesr
p: :	А:А:
АА:А:	А::	А:А:
АА:А:	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	А:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::%!

_output_shapes
:	А:!

_output_shapes	
:А:&	"
 
_output_shapes
:
АА:!


_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::

_output_shapes
: 
╖

ў
C__inference_dense2_layer_call_and_return_conditional_losses_3338422

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
е
Ш
(__inference_dense2_layer_call_fn_3338781

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_33385252
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
В:
·	
#__inference__traced_restore_3338917
file_prefixM
:assignvariableop_twinned_q_network_q_network_dense1_kernel:	АI
:assignvariableop_1_twinned_q_network_q_network_dense1_bias:	АP
<assignvariableop_2_twinned_q_network_q_network_dense2_kernel:
ААI
:assignvariableop_3_twinned_q_network_q_network_dense2_bias:	АO
<assignvariableop_4_twinned_q_network_q_network_output_kernel:	АH
:assignvariableop_5_twinned_q_network_q_network_output_bias:Q
>assignvariableop_6_twinned_q_network_q_network_1_dense1_kernel:	АK
<assignvariableop_7_twinned_q_network_q_network_1_dense1_bias:	АR
>assignvariableop_8_twinned_q_network_q_network_1_dense2_kernel:
ААK
<assignvariableop_9_twinned_q_network_q_network_1_dense2_bias:	АR
?assignvariableop_10_twinned_q_network_q_network_1_output_kernel:	АK
=assignvariableop_11_twinned_q_network_q_network_1_output_bias:
identity_13ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Й
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Х
valueЛBИB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesи
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesь
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

Identity╣
AssignVariableOpAssignVariableOp:assignvariableop_twinned_q_network_q_network_dense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1┐
AssignVariableOp_1AssignVariableOp:assignvariableop_1_twinned_q_network_q_network_dense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2┴
AssignVariableOp_2AssignVariableOp<assignvariableop_2_twinned_q_network_q_network_dense2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3┐
AssignVariableOp_3AssignVariableOp:assignvariableop_3_twinned_q_network_q_network_dense2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4┴
AssignVariableOp_4AssignVariableOp<assignvariableop_4_twinned_q_network_q_network_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5┐
AssignVariableOp_5AssignVariableOp:assignvariableop_5_twinned_q_network_q_network_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6├
AssignVariableOp_6AssignVariableOp>assignvariableop_6_twinned_q_network_q_network_1_dense1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7┴
AssignVariableOp_7AssignVariableOp<assignvariableop_7_twinned_q_network_q_network_1_dense1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8├
AssignVariableOp_8AssignVariableOp>assignvariableop_8_twinned_q_network_q_network_1_dense2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9┴
AssignVariableOp_9AssignVariableOp<assignvariableop_9_twinned_q_network_q_network_1_dense2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10╟
AssignVariableOp_10AssignVariableOp?assignvariableop_10_twinned_q_network_q_network_1_output_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11┼
AssignVariableOp_11AssignVariableOp=assignvariableop_11_twinned_q_network_q_network_1_output_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpц
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12┘
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
┘
Н
-__inference_q_network_1_layer_call_fn_3338566
input_1
unknown:	А
	unknown_0:	А
	unknown_1:
АА
	unknown_2:	А
	unknown_3:	А
	unknown_4:
identityИвStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_q_network_1_layer_call_and_return_conditional_losses_33385482
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
╖

ў
C__inference_dense2_layer_call_and_return_conditional_losses_3338733

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Л
╛
%__inference_signature_wrapper_3338693
input_1
unknown:	А
	unknown_0:	А
	unknown_1:
АА
	unknown_2:	А
	unknown_3:	А
	unknown_4:
	unknown_5:	А
	unknown_6:	А
	unknown_7:
АА
	unknown_8:	А
	unknown_9:	А

unknown_10:
identity

identity_1ИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         *.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *+
f&R$
"__inference__wrapped_model_33383902
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
╙	
ї
C__inference_output_layer_call_and_return_conditional_losses_3338811

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╖

ў
C__inference_dense2_layer_call_and_return_conditional_losses_3338525

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╖

ў
C__inference_dense2_layer_call_and_return_conditional_losses_3338792

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
б
Ц
(__inference_output_layer_call_fn_3338801

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_33385412
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
│

Ў
C__inference_dense1_layer_call_and_return_conditional_losses_3338508

inputs1
matmul_readvariableop_resource:	А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
│

Ў
C__inference_dense1_layer_call_and_return_conditional_losses_3338405

inputs1
matmul_readvariableop_resource:	А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*щ
serving_default╒
;
input_10
serving_default_input_1:0         <
output_10
StatefulPartitionedCall:0         <
output_20
StatefulPartitionedCall:1         tensorflow/serving/predict:а╖
К
Q1
Q2
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
j__call__
k_default_save_signature
*l&call_and_return_all_conditional_losses"п
_tf_keras_modelХ{"name": "twinned_q_network", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "TwinnedQNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 23]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "TwinnedQNetwork"}}
█
	hidden_units


dense1

dense2
out
	variables
regularization_losses
trainable_variables
	keras_api
m__call__
*n&call_and_return_all_conditional_losses"Щ
_tf_keras_model {"name": "q_network", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "QNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 23]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "QNetwork"}}
▌
hidden_units

dense1

dense2
out
	variables
regularization_losses
trainable_variables
	keras_api
o__call__
*p&call_and_return_all_conditional_losses"Ы
_tf_keras_modelБ{"name": "q_network_1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "QNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 23]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "QNetwork"}}
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
╩
	variables
%layer_metrics
&metrics
'layer_regularization_losses

(layers
regularization_losses
)non_trainable_variables
trainable_variables
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
╚

kernel
bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
r__call__
*s&call_and_return_all_conditional_losses"г
_tf_keras_layerЙ{"name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 23}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 23]}}
╩

kernel
bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
t__call__
*u&call_and_return_all_conditional_losses"е
_tf_keras_layerЛ{"name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
╠

kernel
bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
v__call__
*w&call_and_return_all_conditional_losses"з
_tf_keras_layerН{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float64", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
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
н
	variables
6layer_metrics
7metrics
8layer_regularization_losses

9layers
regularization_losses
:non_trainable_variables
trainable_variables
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
╠

kernel
 bias
;	variables
<regularization_losses
=trainable_variables
>	keras_api
x__call__
*y&call_and_return_all_conditional_losses"з
_tf_keras_layerН{"name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 23}}, "shared_object_id": 15}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 23]}}
╬

!kernel
"bias
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
z__call__
*{&call_and_return_all_conditional_losses"й
_tf_keras_layerП{"name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
╬

#kernel
$bias
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
|__call__
*}&call_and_return_all_conditional_losses"й
_tf_keras_layerП{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float64", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
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
н
	variables
Glayer_metrics
Hmetrics
Ilayer_regularization_losses

Jlayers
regularization_losses
Knon_trainable_variables
trainable_variables
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
<::	А2)twinned_q_network/q_network/dense1/kernel
6:4А2'twinned_q_network/q_network/dense1/bias
=:;
АА2)twinned_q_network/q_network/dense2/kernel
6:4А2'twinned_q_network/q_network/dense2/bias
<::	А2)twinned_q_network/q_network/output/kernel
5:32'twinned_q_network/q_network/output/bias
>:<	А2+twinned_q_network/q_network_1/dense1/kernel
8:6А2)twinned_q_network/q_network_1/dense1/bias
?:=
АА2+twinned_q_network/q_network_1/dense2/kernel
8:6А2)twinned_q_network/q_network_1/dense2/bias
>:<	А2+twinned_q_network/q_network_1/output/kernel
7:52)twinned_q_network/q_network_1/output/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
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
н
*	variables
Llayer_metrics
Mmetrics
Nlayer_regularization_losses

Olayers
+regularization_losses
Pnon_trainable_variables
,trainable_variables
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
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
н
.	variables
Qlayer_metrics
Rmetrics
Slayer_regularization_losses

Tlayers
/regularization_losses
Unon_trainable_variables
0trainable_variables
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
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
н
2	variables
Vlayer_metrics
Wmetrics
Xlayer_regularization_losses

Ylayers
3regularization_losses
Znon_trainable_variables
4trainable_variables
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
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
н
;	variables
[layer_metrics
\metrics
]layer_regularization_losses

^layers
<regularization_losses
_non_trainable_variables
=trainable_variables
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
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
н
?	variables
`layer_metrics
ametrics
blayer_regularization_losses

clayers
@regularization_losses
dnon_trainable_variables
Atrainable_variables
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
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
н
C	variables
elayer_metrics
fmetrics
glayer_regularization_losses

hlayers
Dregularization_losses
inon_trainable_variables
Etrainable_variables
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
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
№2∙
3__inference_twinned_q_network_layer_call_fn_3338660┴
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *&в#
!К
input_1         
р2▌
"__inference__wrapped_model_3338390╢
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *&в#
!К
input_1         
Ч2Ф
N__inference_twinned_q_network_layer_call_and_return_conditional_losses_3338628┴
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *&в#
!К
input_1         
Ї2ё
+__inference_q_network_layer_call_fn_3338463┴
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *&в#
!К
input_1         
П2М
F__inference_q_network_layer_call_and_return_conditional_losses_3338445┴
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *&в#
!К
input_1         
Ў2є
-__inference_q_network_1_layer_call_fn_3338566┴
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *&в#
!К
input_1         
С2О
H__inference_q_network_1_layer_call_and_return_conditional_losses_3338548┴
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *&в#
!К
input_1         
╠B╔
%__inference_signature_wrapper_3338693input_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense1_layer_call_fn_3338702в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense1_layer_call_and_return_conditional_losses_3338713в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense2_layer_call_fn_3338722в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense2_layer_call_and_return_conditional_losses_3338733в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_output_layer_call_fn_3338742в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_output_layer_call_and_return_conditional_losses_3338752в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense1_layer_call_fn_3338761в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense1_layer_call_and_return_conditional_losses_3338772в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense2_layer_call_fn_3338781в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense2_layer_call_and_return_conditional_losses_3338792в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_output_layer_call_fn_3338801в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_output_layer_call_and_return_conditional_losses_3338811в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ╠
"__inference__wrapped_model_3338390е !"#$0в-
&в#
!К
input_1         
к "cк`
.
output_1"К
output_1         
.
output_2"К
output_2         д
C__inference_dense1_layer_call_and_return_conditional_losses_3338713]/в,
%в"
 К
inputs         
к "&в#
К
0         А
Ъ д
C__inference_dense1_layer_call_and_return_conditional_losses_3338772] /в,
%в"
 К
inputs         
к "&в#
К
0         А
Ъ |
(__inference_dense1_layer_call_fn_3338702P/в,
%в"
 К
inputs         
к "К         А|
(__inference_dense1_layer_call_fn_3338761P /в,
%в"
 К
inputs         
к "К         Ае
C__inference_dense2_layer_call_and_return_conditional_losses_3338733^0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ е
C__inference_dense2_layer_call_and_return_conditional_losses_3338792^!"0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ }
(__inference_dense2_layer_call_fn_3338722Q0в-
&в#
!К
inputs         А
к "К         А}
(__inference_dense2_layer_call_fn_3338781Q!"0в-
&в#
!К
inputs         А
к "К         Ад
C__inference_output_layer_call_and_return_conditional_losses_3338752]0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ д
C__inference_output_layer_call_and_return_conditional_losses_3338811]#$0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ |
(__inference_output_layer_call_fn_3338742P0в-
&в#
!К
inputs         А
к "К         |
(__inference_output_layer_call_fn_3338801P#$0в-
&в#
!К
inputs         А
к "К         н
H__inference_q_network_1_layer_call_and_return_conditional_losses_3338548a !"#$0в-
&в#
!К
input_1         
к "%в"
К
0         
Ъ Е
-__inference_q_network_1_layer_call_fn_3338566T !"#$0в-
&в#
!К
input_1         
к "К         л
F__inference_q_network_layer_call_and_return_conditional_losses_3338445a0в-
&в#
!К
input_1         
к "%в"
К
0         
Ъ Г
+__inference_q_network_layer_call_fn_3338463T0в-
&в#
!К
input_1         
к "К         ┌
%__inference_signature_wrapper_3338693░ !"#$;в8
в 
1к.
,
input_1!К
input_1         "cк`
.
output_1"К
output_1         
.
output_2"К
output_2         р
N__inference_twinned_q_network_layer_call_and_return_conditional_losses_3338628Н !"#$0в-
&в#
!К
input_1         
к "KвH
Aв>
К
0/0         
К
0/1         
Ъ ╢
3__inference_twinned_q_network_layer_call_fn_3338660 !"#$0в-
&в#
!К
input_1         
к "=в:
К
0         
К
1         