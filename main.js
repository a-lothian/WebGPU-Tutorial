let PIXELS_PER_CELL = 4;
let GRID_SIZE_X = 64;
let GRID_SIZE_Y = 64;
let TARGET_SIM_RATE = 1;
let BRUSH_RADIUS_SQRED = 8; // brush radius squared
const WORKGROUP_SIZE = 8;
let step = 0; // count number of frames simulated
let simAcc = 0; // accumulator to allow fractional simulation speeds

const canvas = document.querySelector("canvas");

const resInputX = document.querySelector("#res-input-x");
const resMeterX = document.querySelector("#res-meter-x");

const resInputY = document.querySelector("#res-input-y");
const resMeterY = document.querySelector("#res-meter-y");

const cellScale = document.querySelector("#cell-size");
const cellScaleMeter = document.querySelector("#cell-size-meter");

const simSpeed = document.querySelector("#sim-speed");
const simSpeedMeter = document.querySelector("#sim-speed-meter");

const brushSize = document.querySelector("#brush-size");
const brushSizeMeter = document.querySelector("#brush-size-meter");

const fpsMeter = document.querySelector("#fps-meter");

let lastFrameTime = performance.now();

let isDragging = false;

const cellsToAdd = new Set();
const singleCellValue = new Uint32Array([1]);

// check WebGPU support
if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
}

// check for compatible hardware
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
}

// get interface with hardware
const device = await adapter.requestDevice();

// configure + link canvas
const context = canvas.getContext("webgpu");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: canvasFormat,
});

device.onuncapturederror = (event) => {
    console.error("WebGPU error:", event.error);
};

// interface for providing GPU instructions
// const encoder = device.createCommandEncoder();

// define geometry to render
const vertices = new Float32Array([
    // X     Y
    -1, -1,
    1, -1,    // Triangle 1
    1, 1,

    -1, -1,
    1, 1,    // Triangle 2
    -1, 1,
]);

// create GPU buffer to store vertices on the GPU side
const vertexBuffer = device.createBuffer({
    label: "Cell vertices",     // custom label used for debugging
    size: vertices.byteLength,  // Float32Array provides length attribute
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});

// copy vertex data into GPU buffer
device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/ 0, vertices);

// define layout of the buffer
const vertexBufferLayout = {
    arrayStride: 8, // each coord contains 2 floats of 32 bit (4 bytes) = 8 bytes per coordinate
    attributes: [{
        format: "float32x2", // 2D coordinates of 32 bit floats
        offset: 0,  // 1st attribute, so data starts at 0. increases if > 1 attribute within buffer
        shaderLocation: 0, // Position, see vertex shader
    }],
};

// uniform buffers are constant across all computation each frame; they dont change based on position, etc.
// buffer is set using javascript before rendering, then is read-only during draw calls
const uniformArray = new Float32Array([GRID_SIZE_X, GRID_SIZE_Y]); // store grid resolution in gpu uniform buffer
const uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

device.queue.writeBuffer(uniformBuffer, 0, uniformArray); // write to buffer

const clickPosArray = new Uint32Array([0, GRID_SIZE_X, GRID_SIZE_Y, BRUSH_RADIUS_SQRED]); // store clickStatus, x, y, radius**2 info for spawning new cells with mouse
const clickPosBuffer = device.createBuffer({
    label: "Mouse Info",
    size: clickPosArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
})
device.queue.writeBuffer(clickPosBuffer, 0, clickPosArray);

// Create buffer to track cell state using a STORAGE type
const cellStateArray = new Uint32Array(GRID_SIZE_X * GRID_SIZE_Y);

// create current and working copies of simulation
const cellStateStorage = [
    device.createBuffer({
        label: "Cell State A",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
        label: "Cell State B",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
];

// fill in random cells
for (let i = 0; i < cellStateArray.length; ++i) {
    cellStateArray[i] = Math.random() > 0.6 ? 1 : 0;
}
device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);

const simulationShaderModule = device.createShaderModule({
    label: "Life simulation shader",
    code: /* wgsl */`

        struct MouseInfo {
            click: u32,
            x: u32,
            y: u32,
            radiusSquared: u32,
        };

        @group(0) @binding(0) var<uniform> grid: vec2f;
        @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;
        @group(0) @binding(3) var<uniform> mouseInfo: MouseInfo;
        
        fn cellActive(x: u32, y: u32) -> u32 {
            return cellStateIn[cellIndex(vec2(x, y))];
        }

        fn cellIndex(cell: vec2u) -> u32 {
            return (cell.y % u32(grid.y)) * u32(grid.x) +
            (cell.x % u32(grid.x));
        }

        fn distanceSquared(a: vec2u, b: vec2u) -> u32 {
            let dx = i32(a.x) - i32(b.x);
            let dy = i32(a.y) - i32(b.y);
            return u32(dx*dx + dy*dy);
        }

        fn calculateCell(cell: vec2u) -> u32 {
            let activeNeighbors = cellActive(cell.x+1, cell.y+1) +
                cellActive(cell.x+1, cell.y) +
                cellActive(cell.x+1, cell.y-1) +
                cellActive(cell.x, cell.y-1) +
                cellActive(cell.x-1, cell.y-1) +
                cellActive(cell.x-1, cell.y) +
                cellActive(cell.x-1, cell.y+1) +
                cellActive(cell.x, cell.y+1);
                
            let i = cellIndex(cell.xy);

            switch activeNeighbors {
                case 2: {
                return cellStateIn[i];
                }
                case 3: {
                return 1;
                }
                default: {
                return 0;
                }
            }
        }

        @compute
        @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
            
            let i = cellIndex(cell.xy);

            switch(mouseInfo.click) { // mouse is not clicked
                case 0, default: {
                    cellStateOut[i] = calculateCell(cell.xy);
                }
                
                case 1: { // mouse clicked
                    let cellPos = vec2u(cell.xy);
                    let mousePos = vec2u(mouseInfo.x, mouseInfo.y);
                    let dist = distanceSquared(cellPos, mousePos);
                    let radius = mouseInfo.radiusSquared;

                    if (dist <= radius) {
                        cellStateOut[i] = 1; // cell is within brush
                    } else {
                        cellStateOut[i] = calculateCell(cell.xy);
                    }
                }
            }
        }
    `
})

// defining WGSL shader
const cellShaderModule = device.createShaderModule({
    label: "Cell shader",
    code: /* wgsl */`

        struct VertexInput {
            @location(0) pos: vec2f,
            @builtin(instance_index) instance: u32,
        };

        struct VertexOutput {
            @builtin(position) pos: vec4f,
            @location(0) cell: vec2f,
        };

        @group(0) @binding(0) var<uniform> grid: vec2f;
        @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
        
        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {

            let i = f32(input.instance);
            let cell = vec2f(i % grid.x, floor(i / grid.x));
            let state = f32(cellStateIn[input.instance]);
            let cellOffset = cell / grid * 2;
            let gridPos = (input.pos*state + 1) / grid - 1 + cellOffset;
            
            
            var output: VertexOutput;
            output.pos = vec4f(gridPos, 0, 1);
            output.cell = cell;
            return output;
        }

        @fragment
        fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
            let c = input.cell / grid;
            return vec4f(c, 1-c.x, 1);
        }
        `
});

const bindGroupLayout = device.createBindGroupLayout({
    label: "Cell Bind Group Layout",
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
        buffer: {} // uniform buffer (grid size)
    }, {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
    }, {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
    }, {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {type: "uniform"}
    }]
});

const pipelineLayout = device.createPipelineLayout({
    label: "Cell Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
});

const cellPipeline = device.createRenderPipeline({
    label: "Cell Pipeline",
    layout: pipelineLayout,
    vertex: {
        module: cellShaderModule,     // variable holding the shader functions
        entryPoint: "vertexMain",       // name of function
        buffers: [vertexBufferLayout]   // gpu buffer passed as args
    },
    fragment: {
        module: cellShaderModule,
        entryPoint: "fragmentMain",
        targets: [{
            format: canvasFormat
        }]
    }
});

const simulationPipeline = device.createComputePipeline({
    label: "Simulation pipeline",
    layout: pipelineLayout,
    compute: {
        module: simulationShaderModule,
        entryPoint: "computeMain",
    }
});

// Bindgroups are used to pass arbitrary data 
const bindGroups = [
    device.createBindGroup({
        label: "Cell Renderer Bind Group A",
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        },
        {
            binding: 1,
            resource: { buffer: cellStateStorage[0] }
        },
        {
            binding: 2,
            resource: { buffer: cellStateStorage[1] }
        },
        {
            binding: 3,
            resource: { buffer: clickPosBuffer }
        }
        ],
    }),
    device.createBindGroup({
        label: "Cell Renderer Bind Group B",
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        },
        {
            binding: 1,
            resource: { buffer: cellStateStorage[1] }
        },
        {
            binding: 2,
            resource: { buffer: cellStateStorage[0] }
        },
        {
            binding: 3,
            resource: { buffer: clickPosBuffer }
        }
        ],
    })
];

function updateGrid(inputBufferIndex) {

    // Run compute shader
    const encoder = device.createCommandEncoder();
    const computePass = encoder.beginComputePass();

    computePass.setPipeline(simulationPipeline);
    computePass.setBindGroup(0, bindGroups[inputBufferIndex]);

    const workgroupCountX = Math.ceil(GRID_SIZE_X / WORKGROUP_SIZE);
    const workgroupCountY = Math.ceil(GRID_SIZE_Y / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY);

    computePass.end();
    device.queue.submit([encoder.finish()]);
}

function renderGrid(inputBufferIndex) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: "clear", // clear texture on start of render pass
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            storeOp: "store", // save results of renderpass onto texture 
        }]
    });

    pass.setPipeline(cellPipeline); // specify render pipeline to use
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setBindGroup(0, bindGroups[inputBufferIndex]); // bind uniform buffer
    pass.draw(vertices.length / 2, GRID_SIZE_X * GRID_SIZE_Y); // 6 vertices
    pass.end(); // end of instructions for this render pass
    device.queue.submit([encoder.finish()]);  // submit instruction buffer without storing it instead
}

function renderLoop() {
    const currentTime = performance.now();
    const deltaTime = currentTime - lastFrameTime;
    lastFrameTime = currentTime;

    if (deltaTime > 0) {
        const fps = 1000 / deltaTime;
        fpsMeter.textContent = fps.toFixed(2);
    }

    simAcc += TARGET_SIM_RATE;

    while (simAcc >= 1) {
        updateGrid(step % 2);
        step++;
        simAcc -= 1;
    }

    renderGrid(step % 2);

    requestAnimationFrame(renderLoop);
}

requestAnimationFrame(renderLoop);

function changeSimulationSize() {
    resizeGrid(Number(resInputX.value), Number(resInputY.value));
    canvas.width = GRID_SIZE_X * PIXELS_PER_CELL;
    canvas.height = GRID_SIZE_Y * PIXELS_PER_CELL;
}

resInputX.addEventListener("input", (e) => {
    changeSimulationSize();
})

resInputY.addEventListener("input", (e) => {
    changeSimulationSize();
})

cellScale.addEventListener("input", (e) => {
    PIXELS_PER_CELL = 2 ** cellScale.value;
    cellScaleMeter.textContent = PIXELS_PER_CELL;
    changeSimulationSize();
})

simSpeed.addEventListener("input", (e) => {
    TARGET_SIM_RATE = 2 ** simSpeed.value;
    simSpeedMeter.textContent = TARGET_SIM_RATE;
})

brushSize.addEventListener("input", (e) => {
    BRUSH_RADIUS_SQRED = brushSize.value*brushSize.value;
    brushSizeMeter.textContent = brushSize.value;
})

//report the mouse position on click
canvas.addEventListener("mousedown", (e) => {
    isDragging = true;
    handleMouseInteraction(e);
});

canvas.addEventListener("mousemove", (e) => {
    if (isDragging) {
        handleMouseInteraction(e);
    }
});

canvas.addEventListener("mouseup", () => {
    isDragging = false;
});

canvas.addEventListener("mouseleave", () => {
    isDragging = false;
});

canvas.addEventListener("mouseup", () => {
    clickPosArray[0] = 0;
    device.queue.writeBuffer(clickPosBuffer, 0, clickPosArray);
    isDragging = false;
});

function handleMouseInteraction(event) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const cellX = Math.floor(mouseX / canvas.width * GRID_SIZE_X);
    const cellY = GRID_SIZE_Y - 1 - Math.floor(mouseY / canvas.height * GRID_SIZE_Y); // flipped

    clickPosArray[0] = 1;            // mouse pressed
    clickPosArray[1] = cellX;
    clickPosArray[2] = cellY;
    clickPosArray[3] = BRUSH_RADIUS_SQRED;

    device.queue.writeBuffer(clickPosBuffer, 0, clickPosArray);
}

function coordsToIndex(x, y) {
    const cellX = Math.floor(x / canvas.width * GRID_SIZE_X);
    const cellY = Math.floor(y / canvas.height * GRID_SIZE_Y);

    if (x >= 0 && x < GRID_SIZE_X && y >= 0 && y < GRID_SIZE_Y) {
        return (GRID_SIZE_Y - y - 1) * GRID_SIZE_X + x;
    }
    return null;
}

function resizeGrid(newSizeX, newSizeY) {
    GRID_SIZE_X = newSizeX;
    GRID_SIZE_Y = newSizeY;

    resMeterX.textContent = newSizeX;
    resMeterY.textContent = newSizeY;

    // update gridsize buffer
    device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([newSizeX, newSizeY]));

    // create new buffers with correct sizes
    const cellStateArray = new Uint32Array(GRID_SIZE_X * GRID_SIZE_Y);

    const newCellStateStorage = [
        device.createBuffer({
            label: "Cell State A",
            size: cellStateArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }),
        device.createBuffer({
            label: "Cell State B",
            size: cellStateArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
    ];

    // fill in random cells
    for (let i = 0; i < cellStateArray.length; ++i) {
        cellStateArray[i] = Math.random() > 0.5 ? 1 : 0;
    }
    device.queue.writeBuffer(newCellStateStorage[0], 0, cellStateArray);

    // remake bindGroups pointing to new buffers
    const newBindGroups = [
        device.createBindGroup({
            label: "Cell Renderer Bind Group A",
            layout: bindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: uniformBuffer }
            },
            {
                binding: 1,
                resource: { buffer: newCellStateStorage[0] }
            },
            {
                binding: 2,
                resource: { buffer: newCellStateStorage[1] }
            },
            {
                binding: 3,
                resource: { buffer: clickPosBuffer}
            }
            ],
        }),
        device.createBindGroup({
            label: "Cell Renderer Bind Group B",
            layout: bindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: uniformBuffer }
            },
            {
                binding: 1,
                resource: { buffer: newCellStateStorage[1] }
            },
            {
                binding: 2,
                resource: { buffer: newCellStateStorage[0] }
            },
            {
                binding: 3,
                resource: { buffer: clickPosBuffer}
            }
            ],
        })
    ];

    const prevcellStateStorageA = cellStateStorage[0];
    const prevcellStateStorageB = cellStateStorage[1];

    // update buffer references
    cellStateStorage[0] = newCellStateStorage[0];
    cellStateStorage[1] = newCellStateStorage[1];

    // free vram
    prevcellStateStorageA.destroy();
    prevcellStateStorageB.destroy();

    bindGroups[0] = newBindGroups[0];
    bindGroups[1] = newBindGroups[1];

    cellsToAdd.clear();
    step = 0;
}