??
??
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
2	??
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
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
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

2	?
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
?
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
?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
actor/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_nameactor/dense1/kernel
|
'actor/dense1/kernel/Read/ReadVariableOpReadVariableOpactor/dense1/kernel*
_output_shapes
:	?*
dtype0
{
actor/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameactor/dense1/bias
t
%actor/dense1/bias/Read/ReadVariableOpReadVariableOpactor/dense1/bias*
_output_shapes	
:?*
dtype0
?
actor/dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameactor/dense2/kernel
}
'actor/dense2/kernel/Read/ReadVariableOpReadVariableOpactor/dense2/kernel* 
_output_shapes
:
??*
dtype0
{
actor/dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameactor/dense2/bias
t
%actor/dense2/bias/Read/ReadVariableOpReadVariableOpactor/dense2/bias*
_output_shapes	
:?*
dtype0
?
actor/means/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*#
shared_nameactor/means/kernel
z
&actor/means/kernel/Read/ReadVariableOpReadVariableOpactor/means/kernel*
_output_shapes
:	?*
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
?
actor/log_stdss/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameactor/log_stdss/kernel
?
*actor/log_stdss/kernel/Read/ReadVariableOpReadVariableOpactor/log_stdss/kernel*
_output_shapes
:	?*
dtype0
?
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
N
ConstConst*
_output_shapes
: *
dtype0*
valueB 2      ??
P
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2        

NoOpNoOp
?
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
hidden_units
	optimizer

dense1

dense2
	means
log_stds
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
 
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
 
?
$layer_metrics
%layer_regularization_losses
trainable_variables
	variables
&metrics
	regularization_losses
'non_trainable_variables

(layers
 
QO
VARIABLE_VALUEactor/dense1/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEactor/dense1/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
)layer_metrics
*layer_regularization_losses
trainable_variables
	variables
+metrics
regularization_losses
,non_trainable_variables

-layers
QO
VARIABLE_VALUEactor/dense2/kernel(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEactor/dense2/bias&dense2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
.layer_metrics
/layer_regularization_losses
trainable_variables
	variables
0metrics
regularization_losses
1non_trainable_variables

2layers
OM
VARIABLE_VALUEactor/means/kernel'means/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEactor/means/bias%means/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
3layer_metrics
4layer_regularization_losses
trainable_variables
	variables
5metrics
regularization_losses
6non_trainable_variables

7layers
VT
VARIABLE_VALUEactor/log_stdss/kernel*log_stds/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEactor/log_stdss/bias(log_stds/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
8layer_metrics
9layer_regularization_losses
 trainable_variables
!	variables
:metrics
"regularization_losses
;non_trainable_variables

<layers
 
 
 
 

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
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1actor/dense1/kernelactor/dense1/biasactor/dense2/kernelactor/dense2/biasactor/means/kernelactor/means/biasactor/log_stdss/kernelactor/log_stdss/biasConstConst_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference_signature_wrapper_6224
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'actor/dense1/kernel/Read/ReadVariableOp%actor/dense1/bias/Read/ReadVariableOp'actor/dense2/kernel/Read/ReadVariableOp%actor/dense2/bias/Read/ReadVariableOp&actor/means/kernel/Read/ReadVariableOp$actor/means/bias/Read/ReadVariableOp*actor/log_stdss/kernel/Read/ReadVariableOp(actor/log_stdss/bias/Read/ReadVariableOpConst_2*
Tin
2
*
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
GPU2 *0J 8? *&
f!R
__inference__traced_save_6353
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameactor/dense1/kernelactor/dense1/biasactor/dense2/kernelactor/dense2/biasactor/means/kernelactor/means/biasactor/log_stdss/kernelactor/log_stdss/bias*
Tin
2	*
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
GPU2 *0J 8? *)
f$R"
 __inference__traced_restore_6387ǳ
??
?
__inference__wrapped_model_5990
input_1>
+actor_dense1_matmul_readvariableop_resource:	?;
,actor_dense1_biasadd_readvariableop_resource:	??
+actor_dense2_matmul_readvariableop_resource:
??;
,actor_dense2_biasadd_readvariableop_resource:	?=
*actor_means_matmul_readvariableop_resource:	?9
+actor_means_biasadd_readvariableop_resource:A
.actor_log_stdss_matmul_readvariableop_resource:	?=
/actor_log_stdss_biasadd_readvariableop_resource:
actor_mul_x
actor_add_y
identity

identity_1

identity_2??#actor/dense1/BiasAdd/ReadVariableOp?"actor/dense1/MatMul/ReadVariableOp?#actor/dense2/BiasAdd/ReadVariableOp?"actor/dense2/MatMul/ReadVariableOp?&actor/log_stdss/BiasAdd/ReadVariableOp?%actor/log_stdss/MatMul/ReadVariableOp?"actor/means/BiasAdd/ReadVariableOp?!actor/means/MatMul/ReadVariableOp?
"actor/dense1/MatMul/ReadVariableOpReadVariableOp+actor_dense1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"actor/dense1/MatMul/ReadVariableOp?
actor/dense1/MatMulMatMulinput_1*actor/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/dense1/MatMul?
#actor/dense1/BiasAdd/ReadVariableOpReadVariableOp,actor_dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#actor/dense1/BiasAdd/ReadVariableOp?
actor/dense1/BiasAddBiasAddactor/dense1/MatMul:product:0+actor/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/dense1/BiasAdd?
actor/dense1/ReluReluactor/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
actor/dense1/Relu?
"actor/dense2/MatMul/ReadVariableOpReadVariableOp+actor_dense2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"actor/dense2/MatMul/ReadVariableOp?
actor/dense2/MatMulMatMulactor/dense1/Relu:activations:0*actor/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/dense2/MatMul?
#actor/dense2/BiasAdd/ReadVariableOpReadVariableOp,actor_dense2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#actor/dense2/BiasAdd/ReadVariableOp?
actor/dense2/BiasAddBiasAddactor/dense2/MatMul:product:0+actor/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
actor/dense2/BiasAdd?
actor/dense2/ReluReluactor/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
actor/dense2/Relu?
!actor/means/MatMul/ReadVariableOpReadVariableOp*actor_means_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!actor/means/MatMul/ReadVariableOp?
actor/means/MatMulMatMulactor/dense2/Relu:activations:0)actor/means/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
actor/means/MatMul?
"actor/means/BiasAdd/ReadVariableOpReadVariableOp+actor_means_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"actor/means/BiasAdd/ReadVariableOp?
actor/means/BiasAddBiasAddactor/means/MatMul:product:0*actor/means/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
actor/means/BiasAdd?
%actor/log_stdss/MatMul/ReadVariableOpReadVariableOp.actor_log_stdss_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%actor/log_stdss/MatMul/ReadVariableOp?
actor/log_stdss/MatMulMatMulactor/dense2/Relu:activations:0-actor/log_stdss/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
actor/log_stdss/MatMul?
&actor/log_stdss/BiasAdd/ReadVariableOpReadVariableOp/actor_log_stdss_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&actor/log_stdss/BiasAdd/ReadVariableOp?
actor/log_stdss/BiasAddBiasAdd actor/log_stdss/MatMul:product:0.actor/log_stdss/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
actor/log_stdss/BiasAddq
	actor/ExpExp actor/log_stdss/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
	actor/Exp?
&actor/actor_Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2(
&actor/actor_Normal/sample/sample_shape?
actor/actor_Normal/sample/ShapeShapeactor/means/BiasAdd:output:0*
T0*
_output_shapes
:2!
actor/actor_Normal/sample/Shape?
-actor/actor_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-actor/actor_Normal/sample/strided_slice/stack?
/actor/actor_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/actor/actor_Normal/sample/strided_slice/stack_1?
/actor/actor_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/actor/actor_Normal/sample/strided_slice/stack_2?
'actor/actor_Normal/sample/strided_sliceStridedSlice(actor/actor_Normal/sample/Shape:output:06actor/actor_Normal/sample/strided_slice/stack:output:08actor/actor_Normal/sample/strided_slice/stack_1:output:08actor/actor_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'actor/actor_Normal/sample/strided_slice?
!actor/actor_Normal/sample/Shape_1Shapeactor/Exp:y:0*
T0*
_output_shapes
:2#
!actor/actor_Normal/sample/Shape_1?
/actor/actor_Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/actor/actor_Normal/sample/strided_slice_1/stack?
1actor/actor_Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1actor/actor_Normal/sample/strided_slice_1/stack_1?
1actor/actor_Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1actor/actor_Normal/sample/strided_slice_1/stack_2?
)actor/actor_Normal/sample/strided_slice_1StridedSlice*actor/actor_Normal/sample/Shape_1:output:08actor/actor_Normal/sample/strided_slice_1/stack:output:0:actor/actor_Normal/sample/strided_slice_1/stack_1:output:0:actor/actor_Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2+
)actor/actor_Normal/sample/strided_slice_1?
*actor/actor_Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2,
*actor/actor_Normal/sample/BroadcastArgs/s0?
,actor/actor_Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2.
,actor/actor_Normal/sample/BroadcastArgs/s0_1?
'actor/actor_Normal/sample/BroadcastArgsBroadcastArgs5actor/actor_Normal/sample/BroadcastArgs/s0_1:output:00actor/actor_Normal/sample/strided_slice:output:0*
_output_shapes
:2)
'actor/actor_Normal/sample/BroadcastArgs?
)actor/actor_Normal/sample/BroadcastArgs_1BroadcastArgs,actor/actor_Normal/sample/BroadcastArgs:r0:02actor/actor_Normal/sample/strided_slice_1:output:0*
_output_shapes
:2+
)actor/actor_Normal/sample/BroadcastArgs_1?
)actor/actor_Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2+
)actor/actor_Normal/sample/concat/values_0?
%actor/actor_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%actor/actor_Normal/sample/concat/axis?
 actor/actor_Normal/sample/concatConcatV22actor/actor_Normal/sample/concat/values_0:output:0.actor/actor_Normal/sample/BroadcastArgs_1:r0:0.actor/actor_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 actor/actor_Normal/sample/concat?
3actor/actor_Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB 2        25
3actor/actor_Normal/sample/normal/random_normal/mean?
5actor/actor_Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB 2      ??27
5actor/actor_Normal/sample/normal/random_normal/stddev?
Cactor/actor_Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal)actor/actor_Normal/sample/concat:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02E
Cactor/actor_Normal/sample/normal/random_normal/RandomStandardNormal?
2actor/actor_Normal/sample/normal/random_normal/mulMulLactor/actor_Normal/sample/normal/random_normal/RandomStandardNormal:output:0>actor/actor_Normal/sample/normal/random_normal/stddev:output:0*
T0*4
_output_shapes"
 :??????????????????24
2actor/actor_Normal/sample/normal/random_normal/mul?
.actor/actor_Normal/sample/normal/random_normalAdd6actor/actor_Normal/sample/normal/random_normal/mul:z:0<actor/actor_Normal/sample/normal/random_normal/mean:output:0*
T0*4
_output_shapes"
 :??????????????????20
.actor/actor_Normal/sample/normal/random_normal?
actor/actor_Normal/sample/mulMul2actor/actor_Normal/sample/normal/random_normal:z:0actor/Exp:y:0*
T0*+
_output_shapes
:?????????2
actor/actor_Normal/sample/mul?
actor/actor_Normal/sample/addAddV2!actor/actor_Normal/sample/mul:z:0actor/means/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
actor/actor_Normal/sample/add?
!actor/actor_Normal/sample/Shape_2Shape!actor/actor_Normal/sample/add:z:0*
T0*
_output_shapes
:2#
!actor/actor_Normal/sample/Shape_2?
/actor/actor_Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/actor/actor_Normal/sample/strided_slice_2/stack?
1actor/actor_Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1actor/actor_Normal/sample/strided_slice_2/stack_1?
1actor/actor_Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1actor/actor_Normal/sample/strided_slice_2/stack_2?
)actor/actor_Normal/sample/strided_slice_2StridedSlice*actor/actor_Normal/sample/Shape_2:output:08actor/actor_Normal/sample/strided_slice_2/stack:output:0:actor/actor_Normal/sample/strided_slice_2/stack_1:output:0:actor/actor_Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2+
)actor/actor_Normal/sample/strided_slice_2?
'actor/actor_Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'actor/actor_Normal/sample/concat_1/axis?
"actor/actor_Normal/sample/concat_1ConcatV2/actor/actor_Normal/sample/sample_shape:output:02actor/actor_Normal/sample/strided_slice_2:output:00actor/actor_Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"actor/actor_Normal/sample/concat_1?
!actor/actor_Normal/sample/ReshapeReshape!actor/actor_Normal/sample/add:z:0+actor/actor_Normal/sample/concat_1:output:0*
T0*'
_output_shapes
:?????????2#
!actor/actor_Normal/sample/Reshape~

actor/TanhTanh*actor/actor_Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????2

actor/Tanhl
	actor/mulMulactor_mul_xactor/Tanh:y:0*
T0*'
_output_shapes
:?????????2
	actor/mulm
	actor/addAddV2actor/mul:z:0actor_add_y*
T0*'
_output_shapes
:?????????2
	actor/add?
%actor/actor_Normal_1/log_prob/truedivRealDiv*actor/actor_Normal/sample/Reshape:output:0actor/Exp:y:0*
T0*'
_output_shapes
:?????????2'
%actor/actor_Normal_1/log_prob/truediv?
'actor/actor_Normal_1/log_prob/truediv_1RealDivactor/means/BiasAdd:output:0actor/Exp:y:0*
T0*'
_output_shapes
:?????????2)
'actor/actor_Normal_1/log_prob/truediv_1?
/actor/actor_Normal_1/log_prob/SquaredDifferenceSquaredDifference)actor/actor_Normal_1/log_prob/truediv:z:0+actor/actor_Normal_1/log_prob/truediv_1:z:0*
T0*'
_output_shapes
:?????????21
/actor/actor_Normal_1/log_prob/SquaredDifference?
#actor/actor_Normal_1/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB 2      ??2%
#actor/actor_Normal_1/log_prob/mul/x?
!actor/actor_Normal_1/log_prob/mulMul,actor/actor_Normal_1/log_prob/mul/x:output:03actor/actor_Normal_1/log_prob/SquaredDifference:z:0*
T0*'
_output_shapes
:?????????2#
!actor/actor_Normal_1/log_prob/mul?
#actor/actor_Normal_1/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB 2??d??g??2%
#actor/actor_Normal_1/log_prob/Const?
!actor/actor_Normal_1/log_prob/LogLogactor/Exp:y:0*
T0*'
_output_shapes
:?????????2#
!actor/actor_Normal_1/log_prob/Log?
!actor/actor_Normal_1/log_prob/addAddV2,actor/actor_Normal_1/log_prob/Const:output:0%actor/actor_Normal_1/log_prob/Log:y:0*
T0*'
_output_shapes
:?????????2#
!actor/actor_Normal_1/log_prob/add?
!actor/actor_Normal_1/log_prob/subSub%actor/actor_Normal_1/log_prob/mul:z:0%actor/actor_Normal_1/log_prob/add:z:0*
T0*'
_output_shapes
:?????????2#
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
:?????????2
	actor/powc
actor/sub/xConst*
_output_shapes
: *
dtype0*
valueB 2      ??2
actor/sub/xt
	actor/subSubactor/sub/x:output:0actor/pow:z:0*
T0*'
_output_shapes
:?????????2
	actor/subg
actor/add_1/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
actor/add_1/y|
actor/add_1AddV2actor/sub:z:0actor/add_1/y:output:0*
T0*'
_output_shapes
:?????????2
actor/add_1`
	actor/LogLogactor/add_1:z:0*
T0*'
_output_shapes
:?????????2
	actor/Log?
actor/sub_1Sub%actor/actor_Normal_1/log_prob/sub:z:0actor/Log:y:0*
T0*'
_output_shapes
:?????????2
actor/sub_1|
actor/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
actor/Sum/reduction_indices?
	actor/SumSumactor/sub_1:z:0$actor/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
	actor/Sumc
	actor/NegNegactor/Sum:output:0*
T0*'
_output_shapes
:?????????2
	actor/Negt
actor/Tanh_1Tanhactor/means/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
actor/Tanh_1?
IdentityIdentityactor/add:z:0$^actor/dense1/BiasAdd/ReadVariableOp#^actor/dense1/MatMul/ReadVariableOp$^actor/dense2/BiasAdd/ReadVariableOp#^actor/dense2/MatMul/ReadVariableOp'^actor/log_stdss/BiasAdd/ReadVariableOp&^actor/log_stdss/MatMul/ReadVariableOp#^actor/means/BiasAdd/ReadVariableOp"^actor/means/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identityactor/Neg:y:0$^actor/dense1/BiasAdd/ReadVariableOp#^actor/dense1/MatMul/ReadVariableOp$^actor/dense2/BiasAdd/ReadVariableOp#^actor/dense2/MatMul/ReadVariableOp'^actor/log_stdss/BiasAdd/ReadVariableOp&^actor/log_stdss/MatMul/ReadVariableOp#^actor/means/BiasAdd/ReadVariableOp"^actor/means/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identityactor/Tanh_1:y:0$^actor/dense1/BiasAdd/ReadVariableOp#^actor/dense1/MatMul/ReadVariableOp$^actor/dense2/BiasAdd/ReadVariableOp#^actor/dense2/MatMul/ReadVariableOp'^actor/log_stdss/BiasAdd/ReadVariableOp&^actor/log_stdss/MatMul/ReadVariableOp#^actor/means/BiasAdd/ReadVariableOp"^actor/means/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2J
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
:?????????
!
_user_specified_name	input_1:	

_output_shapes
: :


_output_shapes
: 
?	
?
?__inference_means_layer_call_and_return_conditional_losses_6038

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
?
$__inference_actor_layer_call_fn_6153
input_1
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
	unknown_7
	unknown_8
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_actor_layer_call_and_return_conditional_losses_61232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:	

_output_shapes
: :


_output_shapes
: 
?

?
@__inference_dense1_layer_call_and_return_conditional_losses_6235

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
@__inference_dense2_layer_call_and_return_conditional_losses_6255

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
?
?
(__inference_log_stdss_layer_call_fn_6302

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_log_stdss_layer_call_and_return_conditional_losses_60542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
@__inference_dense1_layer_call_and_return_conditional_losses_6005

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_dense2_layer_call_fn_6264

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
GPU2 *0J 8? *I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_60222
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
C__inference_log_stdss_layer_call_and_return_conditional_losses_6293

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
?
__inference__traced_save_6353
file_prefix2
.savev2_actor_dense1_kernel_read_readvariableop0
,savev2_actor_dense1_bias_read_readvariableop2
.savev2_actor_dense2_kernel_read_readvariableop0
,savev2_actor_dense2_bias_read_readvariableop1
-savev2_actor_means_kernel_read_readvariableop/
+savev2_actor_means_bias_read_readvariableop5
1savev2_actor_log_stdss_kernel_read_readvariableop3
/savev2_actor_log_stdss_bias_read_readvariableop
savev2_const_2

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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB'means/kernel/.ATTRIBUTES/VARIABLE_VALUEB%means/bias/.ATTRIBUTES/VARIABLE_VALUEB*log_stds/kernel/.ATTRIBUTES/VARIABLE_VALUEB(log_stds/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_actor_dense1_kernel_read_readvariableop,savev2_actor_dense1_bias_read_readvariableop.savev2_actor_dense2_kernel_read_readvariableop,savev2_actor_dense2_bias_read_readvariableop-savev2_actor_means_kernel_read_readvariableop+savev2_actor_means_bias_read_readvariableop1savev2_actor_log_stdss_kernel_read_readvariableop/savev2_actor_log_stdss_bias_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
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

identity_1Identity_1:output:0*^
_input_shapesM
K: :	?:?:
??:?:	?::	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::	

_output_shapes
: 
?
?
%__inference_dense1_layer_call_fn_6244

inputs
unknown:	?
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
GPU2 *0J 8? *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_60052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_6224
input_1
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
	unknown_7
	unknown_8
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8? *(
f#R!
__inference__wrapped_model_59902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:	

_output_shapes
: :


_output_shapes
: 
?	
?
?__inference_means_layer_call_and_return_conditional_losses_6274

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
@__inference_dense2_layer_call_and_return_conditional_losses_6022

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
?b
?
?__inference_actor_layer_call_and_return_conditional_losses_6123
input_1
dense1_6006:	?
dense1_6008:	?
dense2_6023:
??
dense2_6025:	?

means_6039:	?

means_6041:!
log_stdss_6055:	?
log_stdss_6057:	
mul_x	
add_y
identity

identity_1

identity_2??dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?!log_stdss/StatefulPartitionedCall?means/StatefulPartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_6006dense1_6008*
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
GPU2 *0J 8? *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_60052 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_6023dense2_6025*
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
GPU2 *0J 8? *I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_60222 
dense2/StatefulPartitionedCall?
means/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0
means_6039
means_6041*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_means_layer_call_and_return_conditional_losses_60382
means/StatefulPartitionedCall?
!log_stdss/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0log_stdss_6055log_stdss_6057*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_log_stdss_layer_call_and_return_conditional_losses_60542#
!log_stdss/StatefulPartitionedCallo
ExpExp*log_stdss/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Exp
Normal_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2
Normal_1/sample/sample_shape?
Normal_1/sample/ShapeShape&means/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
Normal_1/sample/Shape?
#Normal_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#Normal_1/sample/strided_slice/stack?
%Normal_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Normal_1/sample/strided_slice/stack_1?
%Normal_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Normal_1/sample/strided_slice/stack_2?
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
Normal_1/sample/Shape_1?
%Normal_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Normal_1/sample/strided_slice_1/stack?
'Normal_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Normal_1/sample/strided_slice_1/stack_1?
'Normal_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Normal_1/sample/strided_slice_1/stack_2?
Normal_1/sample/strided_slice_1StridedSlice Normal_1/sample/Shape_1:output:0.Normal_1/sample/strided_slice_1/stack:output:00Normal_1/sample/strided_slice_1/stack_1:output:00Normal_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
Normal_1/sample/strided_slice_1?
 Normal_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2"
 Normal_1/sample/BroadcastArgs/s0?
"Normal_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2$
"Normal_1/sample/BroadcastArgs/s0_1?
Normal_1/sample/BroadcastArgsBroadcastArgs+Normal_1/sample/BroadcastArgs/s0_1:output:0&Normal_1/sample/strided_slice:output:0*
_output_shapes
:2
Normal_1/sample/BroadcastArgs?
Normal_1/sample/BroadcastArgs_1BroadcastArgs"Normal_1/sample/BroadcastArgs:r0:0(Normal_1/sample/strided_slice_1:output:0*
_output_shapes
:2!
Normal_1/sample/BroadcastArgs_1?
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
Normal_1/sample/concat/axis?
Normal_1/sample/concatConcatV2(Normal_1/sample/concat/values_0:output:0$Normal_1/sample/BroadcastArgs_1:r0:0$Normal_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Normal_1/sample/concat?
)Normal_1/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB 2        2+
)Normal_1/sample/normal/random_normal/mean?
+Normal_1/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB 2      ??2-
+Normal_1/sample/normal/random_normal/stddev?
9Normal_1/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal_1/sample/concat:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02;
9Normal_1/sample/normal/random_normal/RandomStandardNormal?
(Normal_1/sample/normal/random_normal/mulMulBNormal_1/sample/normal/random_normal/RandomStandardNormal:output:04Normal_1/sample/normal/random_normal/stddev:output:0*
T0*4
_output_shapes"
 :??????????????????2*
(Normal_1/sample/normal/random_normal/mul?
$Normal_1/sample/normal/random_normalAdd,Normal_1/sample/normal/random_normal/mul:z:02Normal_1/sample/normal/random_normal/mean:output:0*
T0*4
_output_shapes"
 :??????????????????2&
$Normal_1/sample/normal/random_normal?
Normal_1/sample/mulMul(Normal_1/sample/normal/random_normal:z:0Exp:y:0*
T0*+
_output_shapes
:?????????2
Normal_1/sample/mul?
Normal_1/sample/addAddV2Normal_1/sample/mul:z:0&means/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2
Normal_1/sample/addy
Normal_1/sample/Shape_2ShapeNormal_1/sample/add:z:0*
T0*
_output_shapes
:2
Normal_1/sample/Shape_2?
%Normal_1/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%Normal_1/sample/strided_slice_2/stack?
'Normal_1/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'Normal_1/sample/strided_slice_2/stack_1?
'Normal_1/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Normal_1/sample/strided_slice_2/stack_2?
Normal_1/sample/strided_slice_2StridedSlice Normal_1/sample/Shape_2:output:0.Normal_1/sample/strided_slice_2/stack:output:00Normal_1/sample/strided_slice_2/stack_1:output:00Normal_1/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2!
Normal_1/sample/strided_slice_2?
Normal_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Normal_1/sample/concat_1/axis?
Normal_1/sample/concat_1ConcatV2%Normal_1/sample/sample_shape:output:0(Normal_1/sample/strided_slice_2:output:0&Normal_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Normal_1/sample/concat_1?
Normal_1/sample/ReshapeReshapeNormal_1/sample/add:z:0!Normal_1/sample/concat_1:output:0*
T0*'
_output_shapes
:?????????2
Normal_1/sample/Reshapeh
TanhTanh Normal_1/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????2
TanhT
mulMulmul_xTanh:y:0*
T0*'
_output_shapes
:?????????2
mulU
addAddV2mul:z:0add_y*
T0*'
_output_shapes
:?????????2
add?
Normal_2/log_prob/truedivRealDiv Normal_1/sample/Reshape:output:0Exp:y:0*
T0*'
_output_shapes
:?????????2
Normal_2/log_prob/truediv?
Normal_2/log_prob/truediv_1RealDiv&means/StatefulPartitionedCall:output:0Exp:y:0*
T0*'
_output_shapes
:?????????2
Normal_2/log_prob/truediv_1?
#Normal_2/log_prob/SquaredDifferenceSquaredDifferenceNormal_2/log_prob/truediv:z:0Normal_2/log_prob/truediv_1:z:0*
T0*'
_output_shapes
:?????????2%
#Normal_2/log_prob/SquaredDifference{
Normal_2/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB 2      ??2
Normal_2/log_prob/mul/x?
Normal_2/log_prob/mulMul Normal_2/log_prob/mul/x:output:0'Normal_2/log_prob/SquaredDifference:z:0*
T0*'
_output_shapes
:?????????2
Normal_2/log_prob/mul{
Normal_2/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB 2??d??g??2
Normal_2/log_prob/Constp
Normal_2/log_prob/LogLogExp:y:0*
T0*'
_output_shapes
:?????????2
Normal_2/log_prob/Log?
Normal_2/log_prob/addAddV2 Normal_2/log_prob/Const:output:0Normal_2/log_prob/Log:y:0*
T0*'
_output_shapes
:?????????2
Normal_2/log_prob/add?
Normal_2/log_prob/subSubNormal_2/log_prob/mul:z:0Normal_2/log_prob/add:z:0*
T0*'
_output_shapes
:?????????2
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
:?????????2
powW
sub/xConst*
_output_shapes
: *
dtype0*
valueB 2      ??2
sub/x\
subSubsub/x:output:0pow:z:0*
T0*'
_output_shapes
:?????????2
sub[
add_1/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2	
add_1/yd
add_1AddV2sub:z:0add_1/y:output:0*
T0*'
_output_shapes
:?????????2
add_1N
LogLog	add_1:z:0*
T0*'
_output_shapes
:?????????2
Logk
sub_1SubNormal_2/log_prob/sub:z:0Log:y:0*
T0*'
_output_shapes
:?????????2
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
:?????????*
	keep_dims(2
SumQ
NegNegSum:output:0*
T0*'
_output_shapes
:?????????2
Negr
Tanh_1Tanh&means/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Tanh_1?
IdentityIdentityadd:z:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall"^log_stdss/StatefulPartitionedCall^means/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1IdentityNeg:y:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall"^log_stdss/StatefulPartitionedCall^means/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity
Tanh_1:y:0^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall"^log_stdss/StatefulPartitionedCall^means/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2F
!log_stdss/StatefulPartitionedCall!log_stdss/StatefulPartitionedCall2>
means/StatefulPartitionedCallmeans/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:	

_output_shapes
: :


_output_shapes
: 
?	
?
C__inference_log_stdss_layer_call_and_return_conditional_losses_6054

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
?&
?
 __inference__traced_restore_6387
file_prefix7
$assignvariableop_actor_dense1_kernel:	?3
$assignvariableop_1_actor_dense1_bias:	?:
&assignvariableop_2_actor_dense2_kernel:
??3
$assignvariableop_3_actor_dense2_bias:	?8
%assignvariableop_4_actor_means_kernel:	?1
#assignvariableop_5_actor_means_bias:<
)assignvariableop_6_actor_log_stdss_kernel:	?5
'assignvariableop_7_actor_log_stdss_bias:

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB'means/kernel/.ATTRIBUTES/VARIABLE_VALUEB%means/bias/.ATTRIBUTES/VARIABLE_VALUEB*log_stds/kernel/.ATTRIBUTES/VARIABLE_VALUEB(log_stds/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp$assignvariableop_actor_dense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp$assignvariableop_1_actor_dense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp&assignvariableop_2_actor_dense2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp$assignvariableop_3_actor_dense2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_actor_means_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_actor_means_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp)assignvariableop_6_actor_log_stdss_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp'assignvariableop_7_actor_log_stdss_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8?

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
$__inference_means_layer_call_fn_6283

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_means_layer_call_and_return_conditional_losses_60382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
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

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????<
output_20
StatefulPartitionedCall:1?????????<
output_30
StatefulPartitionedCall:2?????????tensorflow/serving/predict:?j
?
hidden_units
	optimizer

dense1

dense2
	means
log_stds
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
*=&call_and_return_all_conditional_losses
>_default_save_signature
?__call__"?
_tf_keras_model?{"name": "actor", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Actor", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 29]}, "float64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Actor"}}
 "
trackable_list_wrapper
"
	optimizer
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*@&call_and_return_all_conditional_losses
A__call__"?
_tf_keras_layer?{"name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 29}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 29]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*B&call_and_return_all_conditional_losses
C__call__"?
_tf_keras_layer?{"name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float64", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*D&call_and_return_all_conditional_losses
E__call__"?
_tf_keras_layer?{"name": "means", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "means", "trainable": true, "dtype": "float64", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
?

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
*F&call_and_return_all_conditional_losses
G__call__"?
_tf_keras_layer?{"name": "log_stdss", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "log_stdss", "trainable": true, "dtype": "float64", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 15}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
$layer_metrics
%layer_regularization_losses
trainable_variables
	variables
&metrics
	regularization_losses
'non_trainable_variables

(layers
?__call__
>_default_save_signature
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
,
Hserving_default"
signature_map
&:$	?2actor/dense1/kernel
 :?2actor/dense1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
)layer_metrics
*layer_regularization_losses
trainable_variables
	variables
+metrics
regularization_losses
,non_trainable_variables

-layers
A__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
':%
??2actor/dense2/kernel
 :?2actor/dense2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
.layer_metrics
/layer_regularization_losses
trainable_variables
	variables
0metrics
regularization_losses
1non_trainable_variables

2layers
C__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
%:#	?2actor/means/kernel
:2actor/means/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
3layer_metrics
4layer_regularization_losses
trainable_variables
	variables
5metrics
regularization_losses
6non_trainable_variables

7layers
E__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
):'	?2actor/log_stdss/kernel
": 2actor/log_stdss/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
8layer_metrics
9layer_regularization_losses
 trainable_variables
!	variables
:metrics
"regularization_losses
;non_trainable_variables

<layers
G__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
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
?2?
?__inference_actor_layer_call_and_return_conditional_losses_6123?
???
FullArgSpec
args?
jself
jstate
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
input_1?????????
?2?
__inference__wrapped_model_5990?
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
input_1?????????
?2?
$__inference_actor_layer_call_fn_6153?
???
FullArgSpec
args?
jself
jstate
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
input_1?????????
?2?
@__inference_dense1_layer_call_and_return_conditional_losses_6235?
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
%__inference_dense1_layer_call_fn_6244?
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
@__inference_dense2_layer_call_and_return_conditional_losses_6255?
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
%__inference_dense2_layer_call_fn_6264?
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
?__inference_means_layer_call_and_return_conditional_losses_6274?
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
$__inference_means_layer_call_fn_6283?
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
C__inference_log_stdss_layer_call_and_return_conditional_losses_6293?
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
(__inference_log_stdss_layer_call_fn_6302?
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
?B?
"__inference_signature_wrapper_6224input_1"?
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
	J
Const
J	
Const_1?
__inference__wrapped_model_5990?
IJ0?-
&?#
!?
input_1?????????
? "???
.
output_1"?
output_1?????????
.
output_2"?
output_2?????????
.
output_3"?
output_3??????????
?__inference_actor_layer_call_and_return_conditional_losses_6123?
IJ0?-
&?#
!?
input_1?????????
? "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
$__inference_actor_layer_call_fn_6153?
IJ0?-
&?#
!?
input_1?????????
? "Z?W
?
0?????????
?
1?????????
?
2??????????
@__inference_dense1_layer_call_and_return_conditional_losses_6235]/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? y
%__inference_dense1_layer_call_fn_6244P/?,
%?"
 ?
inputs?????????
? "????????????
@__inference_dense2_layer_call_and_return_conditional_losses_6255^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? z
%__inference_dense2_layer_call_fn_6264Q0?-
&?#
!?
inputs??????????
? "????????????
C__inference_log_stdss_layer_call_and_return_conditional_losses_6293]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_log_stdss_layer_call_fn_6302P0?-
&?#
!?
inputs??????????
? "???????????
?__inference_means_layer_call_and_return_conditional_losses_6274]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? x
$__inference_means_layer_call_fn_6283P0?-
&?#
!?
inputs??????????
? "???????????
"__inference_signature_wrapper_6224?
IJ;?8
? 
1?.
,
input_1!?
input_1?????????"???
.
output_1"?
output_1?????????
.
output_2"?
output_2?????????
.
output_3"?
output_3?????????