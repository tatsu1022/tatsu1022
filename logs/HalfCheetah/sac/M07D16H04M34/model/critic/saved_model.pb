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
shape:	*:
shared_name+)twinned_q_network/q_network/dense1/kernel
¨
=twinned_q_network/q_network/dense1/kernel/Read/ReadVariableOpReadVariableOp)twinned_q_network/q_network/dense1/kernel*
_output_shapes
:	*
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
shape:	*<
shared_name-+twinned_q_network/q_network_1/dense1/kernel
¬
?twinned_q_network/q_network_1/dense1/kernel/Read/ReadVariableOpReadVariableOp+twinned_q_network/q_network_1/dense1/kernel*
_output_shapes
:	*
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
regularization_losses
trainable_variables
	variables
	keras_api

signatures

	hidden_units


dense1

dense2
out
regularization_losses
trainable_variables
	variables
	keras_api

hidden_units

dense1

dense2
out
regularization_losses
trainable_variables
	variables
	keras_api
 
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
­
%layer_metrics
regularization_losses
&metrics
trainable_variables

'layers
	variables
(layer_regularization_losses
)non_trainable_variables
 
 
h

kernel
bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
h

kernel
bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
h

kernel
bias
2regularization_losses
3trainable_variables
4	variables
5	keras_api
 
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
­
6layer_metrics
regularization_losses
7metrics
trainable_variables

8layers
	variables
9layer_regularization_losses
:non_trainable_variables
 
h

kernel
 bias
;regularization_losses
<trainable_variables
=	variables
>	keras_api
h

#kernel
$bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
h

!kernel
"bias
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
 
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
­
Glayer_metrics
regularization_losses
Hmetrics
trainable_variables

Ilayers
	variables
Jlayer_regularization_losses
Knon_trainable_variables
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
­
Llayer_metrics
*regularization_losses
Mmetrics
+trainable_variables

Nlayers
,	variables
Olayer_regularization_losses
Pnon_trainable_variables
 

0
1

0
1
­
Qlayer_metrics
.regularization_losses
Rmetrics
/trainable_variables

Slayers
0	variables
Tlayer_regularization_losses
Unon_trainable_variables
 

0
1

0
1
­
Vlayer_metrics
2regularization_losses
Wmetrics
3trainable_variables

Xlayers
4	variables
Ylayer_regularization_losses
Znon_trainable_variables
 
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
­
[layer_metrics
;regularization_losses
\metrics
<trainable_variables

]layers
=	variables
^layer_regularization_losses
_non_trainable_variables
 

#0
$1

#0
$1
­
`layer_metrics
?regularization_losses
ametrics
@trainable_variables

blayers
A	variables
clayer_regularization_losses
dnon_trainable_variables
 

!0
"1

!0
"1
­
elayer_metrics
Cregularization_losses
fmetrics
Dtrainable_variables

glayers
E	variables
hlayer_regularization_losses
inon_trainable_variables
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
 
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
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
"__inference_signature_wrapper_1323
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
__inference__traced_save_1501
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
 __inference__traced_restore_1547«Ê
`
§
__inference__wrapped_model_1020
input_1T
Atwinned_q_network_q_network_dense1_matmul_readvariableop_resource:	Q
Btwinned_q_network_q_network_dense1_biasadd_readvariableop_resource:	U
Atwinned_q_network_q_network_dense2_matmul_readvariableop_resource:
Q
Btwinned_q_network_q_network_dense2_biasadd_readvariableop_resource:	T
Atwinned_q_network_q_network_output_matmul_readvariableop_resource:	P
Btwinned_q_network_q_network_output_biasadd_readvariableop_resource:V
Ctwinned_q_network_q_network_1_dense1_matmul_readvariableop_resource:	S
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
:	*
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
:	*
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
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2v
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
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¾)
â
__inference__traced_save_1501
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
Þ
ü
E__inference_q_network_1_layer_call_and_return_conditional_losses_1178
input_1
dense1_1139:	
dense1_1141:	
dense2_1156:

dense2_1158:	
output_1172:	
output_1174:
identity¢dense1/StatefulPartitionedCall¢dense2/StatefulPartitionedCall¢output/StatefulPartitionedCall
dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_1139dense1_1141*
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
@__inference_dense1_layer_call_and_return_conditional_losses_11382 
dense1/StatefulPartitionedCall«
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_1156dense2_1158*
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
@__inference_dense2_layer_call_and_return_conditional_losses_11552 
dense2/StatefulPartitionedCallª
output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_1172output_1174*
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
@__inference_output_layer_call_and_return_conditional_losses_11712 
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
÷:
÷	
 __inference__traced_restore_1547
file_prefixM
:assignvariableop_twinned_q_network_q_network_dense1_kernel:	I
:assignvariableop_1_twinned_q_network_q_network_dense1_bias:	O
<assignvariableop_2_twinned_q_network_q_network_output_kernel:	H
:assignvariableop_3_twinned_q_network_q_network_output_bias:P
<assignvariableop_4_twinned_q_network_q_network_dense2_kernel:
I
:assignvariableop_5_twinned_q_network_q_network_dense2_bias:	Q
>assignvariableop_6_twinned_q_network_q_network_1_dense1_kernel:	K
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
è
Ý
K__inference_twinned_q_network_layer_call_and_return_conditional_losses_1258
input_1!
q_network_1230:	
q_network_1232:	"
q_network_1234:

q_network_1236:	!
q_network_1238:	
q_network_1240:#
q_network_1_1243:	
q_network_1_1245:	$
q_network_1_1247:

q_network_1_1249:	#
q_network_1_1251:	
q_network_1_1253:
identity

identity_1¢!q_network/StatefulPartitionedCall¢#q_network_1/StatefulPartitionedCallá
!q_network/StatefulPartitionedCallStatefulPartitionedCallinput_1q_network_1230q_network_1232q_network_1234q_network_1236q_network_1238q_network_1240*
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
C__inference_q_network_layer_call_and_return_conditional_losses_10752#
!q_network/StatefulPartitionedCalló
#q_network_1/StatefulPartitionedCallStatefulPartitionedCallinput_1q_network_1_1243q_network_1_1245q_network_1_1247q_network_1_1249q_network_1_1251q_network_1_1253*
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
E__inference_q_network_1_layer_call_and_return_conditional_losses_11782%
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
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2F
!q_network/StatefulPartitionedCall!q_network/StatefulPartitionedCall2J
#q_network_1/StatefulPartitionedCall#q_network_1/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1


%__inference_dense2_layer_call_fn_1411

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
@__inference_dense2_layer_call_and_return_conditional_losses_11552
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
@__inference_dense1_layer_call_and_return_conditional_losses_1402

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
%__inference_output_layer_call_fn_1372

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
@__inference_output_layer_call_and_return_conditional_losses_10682
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
Ó

*__inference_q_network_1_layer_call_fn_1196
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
E__inference_q_network_1_layer_call_and_return_conditional_losses_11782
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


%__inference_dense1_layer_call_fn_1332

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
@__inference_dense1_layer_call_and_return_conditional_losses_10352
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
°

ó
@__inference_dense1_layer_call_and_return_conditional_losses_1138

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
Ï

(__inference_q_network_layer_call_fn_1093
input_1
unknown:	
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
C__inference_q_network_layer_call_and_return_conditional_losses_10752
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
%__inference_dense2_layer_call_fn_1352

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
@__inference_dense2_layer_call_and_return_conditional_losses_10522
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
@__inference_dense1_layer_call_and_return_conditional_losses_1343

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
Ð	
ò
@__inference_output_layer_call_and_return_conditional_losses_1382

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
Ü
ú
C__inference_q_network_layer_call_and_return_conditional_losses_1075
input_1
dense1_1036:	
dense1_1038:	
dense2_1053:

dense2_1055:	
output_1069:	
output_1071:
identity¢dense1/StatefulPartitionedCall¢dense2/StatefulPartitionedCall¢output/StatefulPartitionedCall
dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_1036dense1_1038*
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
@__inference_dense1_layer_call_and_return_conditional_losses_10352 
dense1/StatefulPartitionedCall«
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_1053dense2_1055*
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
@__inference_dense2_layer_call_and_return_conditional_losses_10522 
dense2/StatefulPartitionedCallª
output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_1069output_1071*
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
@__inference_output_layer_call_and_return_conditional_losses_10682 
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


%__inference_dense1_layer_call_fn_1391

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
@__inference_dense1_layer_call_and_return_conditional_losses_11382
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
Ð	
ò
@__inference_output_layer_call_and_return_conditional_losses_1068

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
@__inference_dense1_layer_call_and_return_conditional_losses_1035

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
´

ô
@__inference_dense2_layer_call_and_return_conditional_losses_1155

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
@__inference_dense2_layer_call_and_return_conditional_losses_1422

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

»
"__inference_signature_wrapper_1323
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
__inference__wrapped_model_10202
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
´

ô
@__inference_dense2_layer_call_and_return_conditional_losses_1052

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
¿
É
0__inference_twinned_q_network_layer_call_fn_1290
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
K__inference_twinned_q_network_layer_call_and_return_conditional_losses_12582
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
%__inference_output_layer_call_fn_1431

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
@__inference_output_layer_call_and_return_conditional_losses_11712
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
Ð	
ò
@__inference_output_layer_call_and_return_conditional_losses_1441

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
@__inference_output_layer_call_and_return_conditional_losses_1171

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
@__inference_dense2_layer_call_and_return_conditional_losses_1363

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
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¨¶

Q1
Q2
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
j_default_save_signature
k__call__
*l&call_and_return_all_conditional_losses"¯
_tf_keras_model{"name": "twinned_q_network", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "TwinnedQNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 23]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "TwinnedQNetwork"}}
Û
	hidden_units


dense1

dense2
out
regularization_losses
trainable_variables
	variables
	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_modelÿ{"name": "q_network", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "QNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 23]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "QNetwork"}}
Ý
hidden_units

dense1

dense2
out
regularization_losses
trainable_variables
	variables
	keras_api
o__call__
*p&call_and_return_all_conditional_losses"
_tf_keras_model{"name": "q_network_1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "QNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 23]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "QNetwork"}}
"
	optimizer
 "
trackable_list_wrapper
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
Ê
%layer_metrics
regularization_losses
&metrics
trainable_variables

'layers
	variables
(layer_regularization_losses
)non_trainable_variables
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
*regularization_losses
+trainable_variables
,	variables
-	keras_api
r__call__
*s&call_and_return_all_conditional_losses"£
_tf_keras_layer{"name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 23}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 23]}}
Ê

kernel
bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
t__call__
*u&call_and_return_all_conditional_losses"¥
_tf_keras_layer{"name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
Ì

kernel
bias
2regularization_losses
3trainable_variables
4	variables
5	keras_api
v__call__
*w&call_and_return_all_conditional_losses"§
_tf_keras_layer{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float64", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
 "
trackable_list_wrapper
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
­
6layer_metrics
regularization_losses
7metrics
trainable_variables

8layers
	variables
9layer_regularization_losses
:non_trainable_variables
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
Ì

kernel
 bias
;regularization_losses
<trainable_variables
=	variables
>	keras_api
x__call__
*y&call_and_return_all_conditional_losses"§
_tf_keras_layer{"name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 23}}, "shared_object_id": 15}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 23]}}
Î

#kernel
$bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
z__call__
*{&call_and_return_all_conditional_losses"©
_tf_keras_layer{"name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
Î

!kernel
"bias
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
|__call__
*}&call_and_return_all_conditional_losses"©
_tf_keras_layer{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float64", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
 "
trackable_list_wrapper
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
­
Glayer_metrics
regularization_losses
Hmetrics
trainable_variables

Ilayers
	variables
Jlayer_regularization_losses
Knon_trainable_variables
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
<::	2)twinned_q_network/q_network/dense1/kernel
6:42'twinned_q_network/q_network/dense1/bias
<::	2)twinned_q_network/q_network/output/kernel
5:32'twinned_q_network/q_network/output/bias
=:;
2)twinned_q_network/q_network/dense2/kernel
6:42'twinned_q_network/q_network/dense2/bias
>:<	2+twinned_q_network/q_network_1/dense1/kernel
8:62)twinned_q_network/q_network_1/dense1/bias
>:<	2+twinned_q_network/q_network_1/output/kernel
7:52)twinned_q_network/q_network_1/output/bias
?:=
2+twinned_q_network/q_network_1/dense2/kernel
8:62)twinned_q_network/q_network_1/dense2/bias
 "
trackable_dict_wrapper
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
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Llayer_metrics
*regularization_losses
Mmetrics
+trainable_variables

Nlayers
,	variables
Olayer_regularization_losses
Pnon_trainable_variables
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Qlayer_metrics
.regularization_losses
Rmetrics
/trainable_variables

Slayers
0	variables
Tlayer_regularization_losses
Unon_trainable_variables
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Vlayer_metrics
2regularization_losses
Wmetrics
3trainable_variables

Xlayers
4	variables
Ylayer_regularization_losses
Znon_trainable_variables
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
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
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
­
[layer_metrics
;regularization_losses
\metrics
<trainable_variables

]layers
=	variables
^layer_regularization_losses
_non_trainable_variables
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
­
`layer_metrics
?regularization_losses
ametrics
@trainable_variables

blayers
A	variables
clayer_regularization_losses
dnon_trainable_variables
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
­
elayer_metrics
Cregularization_losses
fmetrics
Dtrainable_variables

glayers
E	variables
hlayer_regularization_losses
inon_trainable_variables
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
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
Ý2Ú
__inference__wrapped_model_1020¶
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
ù2ö
0__inference_twinned_q_network_layer_call_fn_1290Á
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
2
K__inference_twinned_q_network_layer_call_and_return_conditional_losses_1258Á
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
ñ2î
(__inference_q_network_layer_call_fn_1093Á
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
2
C__inference_q_network_layer_call_and_return_conditional_losses_1075Á
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
*__inference_q_network_1_layer_call_fn_1196Á
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
E__inference_q_network_1_layer_call_and_return_conditional_losses_1178Á
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
"__inference_signature_wrapper_1323input_1"
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
%__inference_dense1_layer_call_fn_1332¢
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
@__inference_dense1_layer_call_and_return_conditional_losses_1343¢
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
%__inference_dense2_layer_call_fn_1352¢
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
@__inference_dense2_layer_call_and_return_conditional_losses_1363¢
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
%__inference_output_layer_call_fn_1372¢
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
@__inference_output_layer_call_and_return_conditional_losses_1382¢
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
%__inference_dense1_layer_call_fn_1391¢
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
@__inference_dense1_layer_call_and_return_conditional_losses_1402¢
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
%__inference_dense2_layer_call_fn_1411¢
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
@__inference_dense2_layer_call_and_return_conditional_losses_1422¢
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
%__inference_output_layer_call_fn_1431¢
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
@__inference_output_layer_call_and_return_conditional_losses_1441¢
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
__inference__wrapped_model_1020¥ #$!"0¢-
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
@__inference_dense1_layer_call_and_return_conditional_losses_1343]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¡
@__inference_dense1_layer_call_and_return_conditional_losses_1402] /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 y
%__inference_dense1_layer_call_fn_1332P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿy
%__inference_dense1_layer_call_fn_1391P /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¢
@__inference_dense2_layer_call_and_return_conditional_losses_1363^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¢
@__inference_dense2_layer_call_and_return_conditional_losses_1422^#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 z
%__inference_dense2_layer_call_fn_1352Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿz
%__inference_dense2_layer_call_fn_1411Q#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¡
@__inference_output_layer_call_and_return_conditional_losses_1382]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¡
@__inference_output_layer_call_and_return_conditional_losses_1441]!"0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
%__inference_output_layer_call_fn_1372P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿy
%__inference_output_layer_call_fn_1431P!"0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
E__inference_q_network_1_layer_call_and_return_conditional_losses_1178a #$!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_q_network_1_layer_call_fn_1196T #$!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
C__inference_q_network_layer_call_and_return_conditional_losses_1075a0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_q_network_layer_call_fn_1093T0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ×
"__inference_signature_wrapper_1323° #$!";¢8
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
output_2ÿÿÿÿÿÿÿÿÿÝ
K__inference_twinned_q_network_layer_call_and_return_conditional_losses_1258 #$!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 ³
0__inference_twinned_q_network_layer_call_fn_1290 #$!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "=¢:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ