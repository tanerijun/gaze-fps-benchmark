"""
System Information Module

Provides detailed hardware and software information including CPU, GPU, memory, and OS details.
Requires: psutil, py-cpuinfo
"""

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import cpuinfo
import psutil


def get_cpu_info() -> Dict[str, Any]:
    """Get comprehensive CPU information."""
    info = cpuinfo.get_cpu_info()

    cpu_data = {
        "brand": info.get("brand_raw", "Unknown"),
        "architecture": info.get("arch", "Unknown"),
        "bits": info.get("bits", "Unknown"),
        "count_logical": psutil.cpu_count(logical=True),
        "count_physical": psutil.cpu_count(logical=False),
        "max_frequency_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else "N/A",
        "min_frequency_mhz": psutil.cpu_freq().min if psutil.cpu_freq() else "N/A",
        "current_frequency_mhz": psutil.cpu_freq().current
        if psutil.cpu_freq()
        else "N/A",
        "vendor_id": info.get("vendor_id_raw", "Unknown"),
        "model": info.get("model", "Unknown"),
        "family": info.get("family", "Unknown"),
        "stepping": info.get("stepping", "Unknown"),
        "l1_data_cache_size": info.get("l1_data_cache_size", "Unknown"),
        "l1_instruction_cache_size": info.get("l1_instruction_cache_size", "Unknown"),
        "l2_cache_size": info.get("l2_cache_size", "Unknown"),
        "l3_cache_size": info.get("l3_cache_size", "Unknown"),
        "flags": info.get("flags", []),
    }

    # Get per-CPU usage
    cpu_percent_per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
    cpu_data["usage_per_cpu"] = cpu_percent_per_cpu
    cpu_data["usage_average"] = sum(cpu_percent_per_cpu) / len(cpu_percent_per_cpu)

    # CPU times
    cpu_times = psutil.cpu_times()
    cpu_data["cpu_times"] = {
        "user": cpu_times.user,
        "system": cpu_times.system,
        "idle": cpu_times.idle,
    }

    # CPU stats
    cpu_stats = psutil.cpu_stats()
    cpu_data["cpu_stats"] = {
        "ctx_switches": cpu_stats.ctx_switches,
        "interrupts": cpu_stats.interrupts,
        "soft_interrupts": cpu_stats.soft_interrupts,
        "syscalls": cpu_stats.syscalls if hasattr(cpu_stats, "syscalls") else "N/A",
    }

    return cpu_data


def get_memory_info() -> Dict[str, Any]:
    """Get detailed memory information."""
    virtual_mem = psutil.virtual_memory()
    swap_mem = psutil.swap_memory()

    memory_data = {
        "virtual": {
            "total_gb": virtual_mem.total / (1024**3),
            "available_gb": virtual_mem.available / (1024**3),
            "used_gb": virtual_mem.used / (1024**3),
            "free_gb": virtual_mem.free / (1024**3),
            "percent_used": virtual_mem.percent,
            "active_gb": virtual_mem.active / (1024**3)
            if hasattr(virtual_mem, "active")
            else "N/A",
            "inactive_gb": virtual_mem.inactive / (1024**3)
            if hasattr(virtual_mem, "inactive")
            else "N/A",
            "buffers_gb": virtual_mem.buffers / (1024**3)
            if hasattr(virtual_mem, "buffers")
            else "N/A",
            "cached_gb": virtual_mem.cached / (1024**3)
            if hasattr(virtual_mem, "cached")
            else "N/A",
        },
        "swap": {
            "total_gb": swap_mem.total / (1024**3),
            "used_gb": swap_mem.used / (1024**3),
            "free_gb": swap_mem.free / (1024**3),
            "percent_used": swap_mem.percent,
            "sin_gb": swap_mem.sin / (1024**3) if hasattr(swap_mem, "sin") else "N/A",
            "sout_gb": swap_mem.sout / (1024**3)
            if hasattr(swap_mem, "sout")
            else "N/A",
        },
    }

    return memory_data


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information (PyTorch CUDA if available)."""
    gpu_data = {
        "cuda_available": False,
        "cuda_version": "N/A",
        "cudnn_version": "N/A",
        "device_count": 0,
        "devices": [],
    }

    try:
        import torch

        gpu_data["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            gpu_data["cuda_version"] = torch.version.cuda
            gpu_data["cudnn_version"] = torch.backends.cudnn.version()
            gpu_data["device_count"] = torch.cuda.device_count()

            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    "id": i,
                    "name": device_props.name,
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                    "total_memory_gb": device_props.total_memory / (1024**3),
                    "multi_processor_count": device_props.multi_processor_count,
                }

                # Get current memory usage
                try:
                    device_info["memory_allocated_gb"] = torch.cuda.memory_allocated(
                        i
                    ) / (1024**3)
                    device_info["memory_reserved_gb"] = torch.cuda.memory_reserved(
                        i
                    ) / (1024**3)
                except:
                    pass

                gpu_data["devices"].append(device_info)

        # MPS (Apple Silicon) support
        gpu_data["mps_available"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

    except ImportError:
        pass

    return gpu_data


def get_os_info() -> Dict[str, Any]:
    """Get operating system information."""
    os_data = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "hostname": platform.node(),
        "boot_time": datetime.fromtimestamp(psutil.boot_time()).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    # Uptime
    boot_time = psutil.boot_time()
    uptime_seconds = datetime.now().timestamp() - boot_time
    days = int(uptime_seconds // 86400)
    hours = int((uptime_seconds % 86400) // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    os_data["uptime"] = f"{days}d {hours}h {minutes}m"

    return os_data


def get_disk_info() -> Dict[str, Any]:
    """Get disk usage information."""
    disk_data = {"partitions": []}

    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            partition_data = {
                "device": partition.device,
                "mountpoint": partition.mountpoint,
                "fstype": partition.fstype,
                "total_gb": usage.total / (1024**3),
                "used_gb": usage.used / (1024**3),
                "free_gb": usage.free / (1024**3),
                "percent_used": usage.percent,
            }
            disk_data["partitions"].append(partition_data)
        except PermissionError:
            continue

    # Disk I/O
    disk_io = psutil.disk_io_counters()
    if disk_io:
        disk_data["io"] = {
            "read_count": disk_io.read_count,
            "write_count": disk_io.write_count,
            "read_gb": disk_io.read_bytes / (1024**3),
            "write_gb": disk_io.write_bytes / (1024**3),
            "read_time_ms": disk_io.read_time,
            "write_time_ms": disk_io.write_time,
        }

    return disk_data


def get_network_info() -> Dict[str, Any]:
    """Get network information."""
    net_data = {"interfaces": {}}

    # Network interfaces
    net_if_addrs = psutil.net_if_addrs()
    net_if_stats = psutil.net_if_stats()

    for interface, addrs in net_if_addrs.items():
        interface_data = {
            "addresses": [],
            "is_up": False,
            "speed_mbps": 0,
        }

        for addr in addrs:
            interface_data["addresses"].append(
                {
                    "family": str(addr.family),
                    "address": addr.address,
                    "netmask": addr.netmask,
                    "broadcast": addr.broadcast,
                }
            )

        if interface in net_if_stats:
            stats = net_if_stats[interface]
            interface_data["is_up"] = stats.isup
            interface_data["speed_mbps"] = stats.speed
            interface_data["mtu"] = stats.mtu

        net_data["interfaces"][interface] = interface_data

    # Network I/O
    net_io = psutil.net_io_counters()
    if net_io:
        net_data["io"] = {
            "bytes_sent_gb": net_io.bytes_sent / (1024**3),
            "bytes_recv_gb": net_io.bytes_recv / (1024**3),
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "errin": net_io.errin,
            "errout": net_io.errout,
            "dropin": net_io.dropin,
            "dropout": net_io.dropout,
        }

    return net_data


def get_process_info() -> Dict[str, Any]:
    """Get current process information."""
    process = psutil.Process()

    process_data = {
        "pid": process.pid,
        "name": process.name(),
        "status": process.status(),
        "create_time": datetime.fromtimestamp(process.create_time()).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "num_threads": process.num_threads(),
        "cpu_percent": process.cpu_percent(interval=0.1),
        "memory_info": {
            "rss_mb": process.memory_info().rss / (1024**2),
            "vms_mb": process.memory_info().vms / (1024**2),
            "percent": process.memory_percent(),
        },
        "io_counters": None,
    }

    try:
        io_counters = process.io_counters()
        process_data["io_counters"] = {
            "read_count": io_counters.read_count,
            "write_count": io_counters.write_count,
            "read_mb": io_counters.read_bytes / (1024**2),
            "write_mb": io_counters.write_bytes / (1024**2),
        }
    except (PermissionError, AttributeError):
        pass

    return process_data


def print_system_info(verbose: bool = True):
    """Print comprehensive system information."""
    print("\n" + "=" * 80)
    print("SYSTEM INFORMATION".center(80))
    print("=" * 80)

    # OS Information
    print("\n" + "─" * 80)
    print("OPERATING SYSTEM")
    print("─" * 80)
    os_info = get_os_info()
    print(f"System:              {os_info['system']}")
    print(f"Release:             {os_info['release']}")
    print(f"Version:             {os_info['version']}")
    print(f"Machine:             {os_info['machine']}")
    print(f"Processor:           {os_info['processor']}")
    print(f"Platform:            {os_info['platform']}")
    print(f"Hostname:            {os_info['hostname']}")
    print(f"Boot Time:           {os_info['boot_time']}")
    print(f"Uptime:              {os_info['uptime']}")
    print(
        f"Python Version:      {os_info['python_implementation']} {sys.version.split()[0]}"
    )

    # CPU Information
    print("\n" + "─" * 80)
    print("CPU INFORMATION")
    print("─" * 80)
    cpu_info = get_cpu_info()
    print(f"Brand:               {cpu_info['brand']}")
    print(f"Architecture:        {cpu_info['architecture']} ({cpu_info['bits']} bit)")
    print(f"Vendor ID:           {cpu_info['vendor_id']}")
    print(f"Physical Cores:      {cpu_info['count_physical']}")
    print(f"Logical Cores:       {cpu_info['count_logical']}")
    print(
        f"Max Frequency:       {cpu_info['max_frequency_mhz']:.2f} MHz"
        if isinstance(cpu_info["max_frequency_mhz"], (int, float))
        else f"Max Frequency:       {cpu_info['max_frequency_mhz']}"
    )
    print(
        f"Min Frequency:       {cpu_info['min_frequency_mhz']:.2f} MHz"
        if isinstance(cpu_info["min_frequency_mhz"], (int, float))
        else f"Min Frequency:       {cpu_info['min_frequency_mhz']}"
    )
    print(
        f"Current Frequency:   {cpu_info['current_frequency_mhz']:.2f} MHz"
        if isinstance(cpu_info["current_frequency_mhz"], (int, float))
        else f"Current Frequency:   {cpu_info['current_frequency_mhz']}"
    )
    print(f"Average Usage:       {cpu_info['usage_average']:.1f}%")

    if verbose:
        print("\nPer-Core Usage:")
        for i, usage in enumerate(cpu_info["usage_per_cpu"]):
            print(f"  Core {i:2d}:            {usage:5.1f}%")

    print("\nCache Information:")
    print(f"  L1 Data Cache:     {cpu_info['l1_data_cache_size']}")
    print(f"  L1 Inst Cache:     {cpu_info['l1_instruction_cache_size']}")
    print(f"  L2 Cache:          {cpu_info['l2_cache_size']}")
    print(f"  L3 Cache:          {cpu_info['l3_cache_size']}")

    print("\nCPU Statistics:")
    print(f"  Context Switches:  {cpu_info['cpu_stats']['ctx_switches']:,}")
    print(f"  Interrupts:        {cpu_info['cpu_stats']['interrupts']:,}")
    print(f"  Soft Interrupts:   {cpu_info['cpu_stats']['soft_interrupts']:,}")

    if verbose and cpu_info["flags"]:
        print(f"\nCPU Flags ({len(cpu_info['flags'])} total):")
        flags_str = ", ".join(cpu_info["flags"][:20])
        print(f"  {flags_str}...")

    # Memory Information
    print("\n" + "─" * 80)
    print("MEMORY INFORMATION")
    print("─" * 80)
    mem_info = get_memory_info()
    print("Virtual Memory:")
    print(f"  Total:             {mem_info['virtual']['total_gb']:.2f} GB")
    print(f"  Available:         {mem_info['virtual']['available_gb']:.2f} GB")
    print(
        f"  Used:              {mem_info['virtual']['used_gb']:.2f} GB ({mem_info['virtual']['percent_used']:.1f}%)"
    )
    print(f"  Free:              {mem_info['virtual']['free_gb']:.2f} GB")

    if mem_info["virtual"]["active_gb"] != "N/A":
        print(f"  Active:            {mem_info['virtual']['active_gb']:.2f} GB")
    if mem_info["virtual"]["inactive_gb"] != "N/A":
        print(f"  Inactive:          {mem_info['virtual']['inactive_gb']:.2f} GB")
    if mem_info["virtual"]["cached_gb"] != "N/A":
        print(f"  Cached:            {mem_info['virtual']['cached_gb']:.2f} GB")

    print("\nSwap Memory:")
    print(f"  Total:             {mem_info['swap']['total_gb']:.2f} GB")
    print(
        f"  Used:              {mem_info['swap']['used_gb']:.2f} GB ({mem_info['swap']['percent_used']:.1f}%)"
    )
    print(f"  Free:              {mem_info['swap']['free_gb']:.2f} GB")

    # GPU Information
    print("\n" + "─" * 80)
    print("GPU INFORMATION")
    print("─" * 80)
    gpu_info = get_gpu_info()
    print(f"CUDA Available:      {gpu_info['cuda_available']}")

    if gpu_info["cuda_available"]:
        print(f"CUDA Version:        {gpu_info['cuda_version']}")
        print(f"cuDNN Version:       {gpu_info['cudnn_version']}")
        print(f"GPU Device Count:    {gpu_info['device_count']}")

        for device in gpu_info["devices"]:
            print(f"\nGPU Device {device['id']}:")
            print(f"  Name:              {device['name']}")
            print(f"  Compute Cap:       {device['compute_capability']}")
            print(f"  Total Memory:      {device['total_memory_gb']:.2f} GB")
            print(f"  Multiprocessors:   {device['multi_processor_count']}")

            if "memory_allocated_gb" in device:
                print(f"  Memory Allocated:  {device['memory_allocated_gb']:.2f} GB")
                print(f"  Memory Reserved:   {device['memory_reserved_gb']:.2f} GB")

    if gpu_info.get("mps_available"):
        print("\nApple MPS:           Available (Apple Silicon GPU)")

    # Disk Information
    if verbose:
        print("\n" + "─" * 80)
        print("DISK INFORMATION")
        print("─" * 80)
        disk_info = get_disk_info()

        for i, partition in enumerate(disk_info["partitions"]):
            print(f"\nPartition {i + 1}:")
            print(f"  Device:            {partition['device']}")
            print(f"  Mountpoint:        {partition['mountpoint']}")
            print(f"  Filesystem:        {partition['fstype']}")
            print(f"  Total:             {partition['total_gb']:.2f} GB")
            print(
                f"  Used:              {partition['used_gb']:.2f} GB ({partition['percent_used']:.1f}%)"
            )
            print(f"  Free:              {partition['free_gb']:.2f} GB")

        if "io" in disk_info:
            print("\nDisk I/O (since boot):")
            print(f"  Read Count:        {disk_info['io']['read_count']:,}")
            print(f"  Write Count:       {disk_info['io']['write_count']:,}")
            print(f"  Read:              {disk_info['io']['read_gb']:.2f} GB")
            print(f"  Written:           {disk_info['io']['write_gb']:.2f} GB")

    # Process Information
    print("\n" + "─" * 80)
    print("CURRENT PROCESS")
    print("─" * 80)
    process_info = get_process_info()
    print(f"PID:                 {process_info['pid']}")
    print(f"Name:                {process_info['name']}")
    print(f"Status:              {process_info['status']}")
    print(f"Threads:             {process_info['num_threads']}")
    print(f"CPU Usage:           {process_info['cpu_percent']:.1f}%")
    print(f"Memory RSS:          {process_info['memory_info']['rss_mb']:.2f} MB")
    print(f"Memory VMS:          {process_info['memory_info']['vms_mb']:.2f} MB")
    print(f"Memory Percent:      {process_info['memory_info']['percent']:.2f}%")

    print("\n" + "=" * 80 + "\n")


def get_all_info() -> Dict[str, Any]:
    """Get all system information as a dictionary."""
    return {
        "timestamp": datetime.now().isoformat(),
        "os": get_os_info(),
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "gpu": get_gpu_info(),
        "disk": get_disk_info(),
        "network": get_network_info(),
        "process": get_process_info(),
    }


def save_system_info(filepath: str = "system_info.json", verbose: bool = True) -> None:
    """
    Save all system information to a JSON file.

    Args:
        filepath: Path to save the JSON file
        verbose: Whether to print verbose information
    """
    info = get_all_info()

    filepath_obj = Path(filepath)
    filepath_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(info, f, indent=2, default=str)

    if verbose:
        print(f"✓ System information saved to: {filepath}")
        print(f"  File size: {filepath_obj.stat().st_size / 1024:.2f} KB")


if __name__ == "__main__":
    # When run directly, print system information
    import argparse

    parser = argparse.ArgumentParser(description="Display and save system information")
    parser.add_argument(
        "--save",
        type=str,
        help="Save system info to JSON file (e.g., --save system_info.json)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Don't print verbose information",
    )

    args = parser.parse_args()

    verbose = not args.quiet
    print_system_info(verbose=verbose)

    if args.save:
        save_system_info(filepath=args.save, verbose=verbose)
